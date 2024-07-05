from transformers import (
  BitsAndBytesConfig,
  TrainingArguments,
    Trainer,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    AdamW,
    DefaultDataCollator,
    AutoModelForCausalLM,
    AutoTokenizer,
  )
import torch
import torch.nn as nn
from datasets import load_dataset,Dataset
import bitsandbytes as bnb
import os, platform, warnings,sys
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
import random
from peft import LoraConfig,get_peft_model
import ipdb
# from unsloth import FastLanguageModel


class CreateTrainer():
  def __init__(self,args):
    self.args=args
    self._set_training_arguments()

  def _set_training_arguments(self):

    self.training_arguments = TrainingArguments(
      
        ## general settings
        output_dir=self.args.output_dir,
        run_name=self.args.run_name,
        seed=self.args.seed,
        ddp_find_unused_parameters=False,
        do_train=True,
        do_eval=True if self.args.eval else False,
        num_train_epochs=self.args.num_epochs,

        ## training settings
        fp16= True if not torch.cuda.is_bf16_supported() else False,
        bf16= True if torch.cuda.is_bf16_supported() else False,
        save_strategy="steps",
        save_steps=self.args.eval_steps,
        save_total_limit=self.args.save_total_limit,
        logging_steps=self.args.logging_steps,
        report_to=["mlflow"],
        # optim="adamw_torch",

        ## dataloader settings
        per_device_train_batch_size=self.args.train_batch_size,
        per_device_eval_batch_size=self.args.eval_batch_size,
        dataloader_num_workers=self.args.num_workers,
        )


  def create_trainer_sft(self,model,tokenizer,optimizer,scheduler,train_dataset,eval_dataset):
    def formatting_func(examples):
      outputs=[]

      for idx in range(len(examples['text'])):
        outputs.append(f"{tokenizer.bos_token}{examples['text'][idx]}{tokenizer.eos_token}")

      return outputs


    trainer = SFTTrainer(
    args=self.training_arguments,
    model=model,
    tokenizer=tokenizer,
    optimizers=(optimizer,scheduler),
    dataset_text_field="text",
    # formatting_func=formatting_func,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if self.args.eval else None,
    max_seq_length= self.args.max_seq_length,
    )

    return trainer

  def create_trainer_basic(self):
    pass


def load_optimizer_scheduler(model,
                        optimizer_name:str,
                        scheduler_name:str,
                        total_update_steps:int,
                        warmup_ratio:float,
                        lr=1e-5,
                        optimizer_kwargs=None, # ex) {"weight_decay" : 0.01}
                        scheduler_kwargs:dict=None, # ex) {"num_cycles":0.3}s
                        ):

    # decopuled learning rate
    params= [
        {
            "params": [p for n, p in model.named_parameters() if "proj" in n and p.requires_grad],
            "lr": lr,  # higher learning rate for mlp and attention layers
        },
        {
            "params": [p for n, p in model.named_parameters() if "lm_head" in n and p.requires_grad],
            "lr": lr*0.1, # lower learning rate for lmhead
        },
        {
            "params": [p for n, p in model.named_parameters() if "embed_tokens" in n and p.requires_grad],
            "lr": lr*0.1,  # lower learning rate for embedding layer
        },
    ]
    # ipdb.set_trace() # 

    if optimizer_name.lower()=="adamw" or optimizer_name.lower()=="adamw_torch":
        optimizer=AdamW(params=params,
                        **optimizer_kwargs,
                        )

    elif optimizer_name.lower()=="adamw8bit".lower() or optimizer_name.lower()=="adamw_8bit".lower():
        optimizer = bnb.optim.AdamW8bit(params=params,
                        **optimizer_kwargs,
                        )

        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )

    elif optimizer_name.lower()=="pagedadamw8bit" or optimizer_name.lower()=="paged_adamw_8bit":
        optimizer = bnb.optim.PagedAdamW8bit(params=params,
                        **optimizer_kwargs,
                        )

        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )

    if scheduler_name.lower()=="cosine_with_hard_restarts_schedule_with_warmup" or scheduler_name.lower()=="cosine_with_restarts":

      scheduler=get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                  num_warmup_steps=total_update_steps*warmup_ratio,
                                                                  num_training_steps=total_update_steps,
                                                                  **scheduler_kwargs)
    elif scheduler_name.lower()=="cosine_schedule_with_warmup":

      scheduler=get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                  num_warmup_steps=total_update_steps*warmup_ratio,
                                                                  num_training_steps=total_update_steps,
                                                                  **scheduler_kwargs)
                                         
    return optimizer,scheduler



def load_model_tokenizer(base_model_path,
            task_type:str="causal_lm",
            gradient_checkpointing=False,
            flash_attn=True,
            cache_dir="/azurestorage/huggingface_cache/models",
          additional_special_tokens=None,
          pad_token=None,
          pad_token_id=None,
          use_unsloth=False,
          device_map=None,
          ):

  ##################### set model #######################
  if task_type.lower()=="causal_lm":

    if use_unsloth:
      from unsloth import FastLanguageModel
      model, _ = FastLanguageModel.from_pretrained(
                        model_name = base_model_path,
                        dtype=None,
                    )

    else:
      if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            use_cache=False if gradient_checkpointing else True, # use_cache is incompatible with gradient_checkpointing
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="flash_attention_2" if flash_attn else None,
            cache_dir=cache_dir,
        )
      else:
        model = AutoModelForCausalLM.from_pretrained(
          base_model_path,
          trust_remote_code=True,
          use_cache=False if gradient_checkpointing else True, # use_cache is incompatible with gradient_checkpointing
          torch_dtype=torch.bfloat16,
          attn_implementation="flash_attention_2" if flash_attn else None,
          cache_dir=cache_dir,
      )

  elif task_type.lower()=="sequence_classification":
    pass

  elif task_type.lower()=="question_answering":
    pass

  if gradient_checkpointing:
      model.gradient_checkpointing_enable()

  ##################### set tokenizer #######################
  tokenizer = AutoTokenizer.from_pretrained(base_model_path,trust_remote_code=True,cache_dir=cache_dir)
  ###### set special tokens #####
  if additional_special_tokens is not None:
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

  if not tokenizer.pad_token or tokenizer.pad_token==tokenizer.eos_token:
    ## padding with eos_token might make repetiton in inference.

    if tokenizer.unk_token:
      tokenizer.pad_token=tokenizer.unk_token
    else:
      if pad_token is not None:
        tokenizer.pad_token=pad_token
      if pad_token_id is not None:
        tokenizer.pad_token_id=pad_token_id

  tokenizer.padding_side="right"

  if len(tokenizer)!=int(model.config.vocab_size):
      model.resize_token_embeddings(len(tokenizer))
  assert len(tokenizer)==int(model.config.vocab_size) , 'vocab sizes of the tokenizer and the model should be same'


  return model,tokenizer


def get_lora_model(model,use_unsloth=False,**lora_kwargs):

  if use_unsloth:
    model=FastLanguageModel.get_peft_model(model,**lora_kwargs)

  else:
    lora_config=LoraConfig(**lora_kwargs)
    model=get_peft_model(model,lora_config)

  return model