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

import ipdb

from galore_torch import GaLoreAdamW, GaLoreAdamW8bit


class CreateTrainer():
  def __init__(self,args):
    self.args=args
    self._set_training_arguments()

  def _set_training_arguments(self):

    if self.args.enable_galore:
      self.training_arguments = TrainingArguments(
        output_dir= self.args.output_dir,
          # fp16= True,
          bf16= True,
          optim="galore_adamw_8bit",
          optim_args="rank=64, update_proj_gap=50, scale=0.1",
          # optim_args="rank=64, update_proj_gap=100, scale=0.1",
          optim_target_modules=[r".*attn.*", r".*mlp.*"],
          run_name=self.args.run_name,
        ddp_find_unused_parameters=False,
                          )

    else:
      self.training_arguments = TrainingArguments(
        output_dir= self.args.output_dir,
          # fp16= True,
          bf16= True,
          run_name=self.args.run_name,
        ddp_find_unused_parameters=False,
                          )



    self.training_arguments=self.training_arguments.set_dataloader(train_batch_size=self.args.train_batch_size,
                                                             eval_batch_size=self.args.eval_batch_size,
                                                             pin_memory=True,
                                                             num_workers=self.args.num_workers,
                                                             sampler_seed=self.args.seed
                                                            )
    self.training_arguments=self.training_arguments.set_training(
        learning_rate= self.args.learning_rate,
        batch_size=self.args.train_batch_size,
        weight_decay=self.args.weight_decay,
        num_epochs=self.args.num_epochs,
        gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        seed=self.args.seed,
        )
    self.training_arguments=self.training_arguments.set_logging(strategy="steps",steps=self.args.logging_steps,report_to=["mlflow"])
    self.training_arguments=self.training_arguments.set_save(strategy="steps",steps=self.args.eval_steps,total_limit=self.args.save_total_limit)

    if self.args.eval:
          training_arguments=training_arguments.set_evaluate(strategy="steps", batch_size=self.args.eval_batch_size,steps=self.args.eval_steps,delay=0)


  def create_trainer_sft(self,model,tokenizer,optimizer,scheduler,train_dataset,eval_dataset,peft_config=None,response_template=None,data_collator=None):

    if response_template is not None:
      response_template_with_context=f"\n{response_template}\n"
      response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
      data_collator=DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    else:
      if data_collator is None:
        data_collator=DefaultDataCollator()

    if not self.args.enable_galore:
      trainer = SFTTrainer(
      args=self.training_arguments,
      model=model,
      tokenizer=tokenizer,
      optimizers=(optimizer,scheduler),
      train_dataset=train_dataset,
      eval_dataset=eval_dataset if self.args.eval else None,
      data_collator=data_collator,
      peft_config=peft_config,
      max_seq_length= self.args.max_seq_length,
      dataset_text_field=self.args.dataset_text_field if self.args.dataset_text_field else None,
      )
    else:
      trainer = SFTTrainer(
      args=self.training_arguments,
      model=model,
      tokenizer=tokenizer,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset if self.args.eval else None,
      data_collator=data_collator,
      peft_config=peft_config,
      max_seq_length= self.args.max_seq_length,
      dataset_text_field=self.args.dataset_text_field if self.args.dataset_text_field else None,
      )


    return trainer

  def create_trainer_basic(self):
    pass


def load_tokenizer(base_model_path,additional_special_tokens:list=None,pad_token=None,pad_token_id=None):
  """
  base_model_path : path or name of pretrained model
  """
  tokenizer = AutoTokenizer.from_pretrained(base_model_path,trust_remote_code=True)

  ###### set special tokens #####
  if additional_special_tokens is not None:
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

  if not tokenizer.pad_token or tokenizer.pad_token==tokenizer.eos_token:
    ## padding with eos_token might make repetiton in inference.
    if pad_token is not None:
      tokenizer.pad_token=pad_token
    if pad_token_id is not None:
      tokenizer.pad_token_id=pad_token_id

  tokenizer.padding_side="right"
  return tokenizer


def load_model(base_model_path,
            task_type:str="causal_lm",
            gradient_checkpointing=False,
            quantization_config=None,
            flash_attn=True,
            cache_dir="/azurestorage/huggingface_cache/models",
            **kwargs,
            ):

    if task_type.lower()=="causal_lm":
      model = AutoModelForCausalLM.from_pretrained(
          base_model_path,
          trust_remote_code=True,
          use_cache=False if gradient_checkpointing else True, # use_cache is incompatible with gradient_checkpointing
          torch_dtype="auto",
          # device_map="cuda",
          attn_implementation="flash_attention_2" if flash_attn else None,
          cache_dir=cache_dir,
          **kwargs,
      )

    elif task_type.lower()=="sequence_classification":
      pass

    elif task_type.lower()=="question_answering":
      pass

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def load_optimizer_scheduler(model,
                        optimizer_name:str,
                        scheduler_name:str,
                        total_update_steps:int,
                        warmup_ratio:float,
                        optimizer_kwargs:dict=None, # ex) {"lr":1e-5,"weight_decay":0.01}
                        scheduler_kwargs:dict=None, # ex) {"num_cycles":0.3}
                        galore_kwargs:dict=None,
                        ):

    optimizer_name=optimizer_name.lower().strip()
    params=filter(lambda x:x.requires_grad,model.parameters())
                  

    if optimizer_name=="adamw":
        optimizer=AdamW(params=params,
                        **optimizer_kwargs,
                        )

    elif optimizer_name=="adam8bit":
        optimizer = bnb.optim.Adam8bit(params=params,
                        **optimizer_kwargs,
                        )

        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )

    elif optimizer_name=="pagedadam8bit":
        optimizer = bnb.optim.PagedAdam8bit(params=params,
                        **optimizer_kwargs,
                        )

        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )

    elif "galore" in optimizer_name:
        # label layers for galore optimizer
        target_modules_list = ["attn", "mlp"]
        # target_modules_list = ["q_proj", "v_proj"]
        galore_params = []
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            galore_params.append(module.weight)

        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        
        galore_param_dict={'params': galore_params}
        galore_param_dict.update(galore_kwargs)

        param_groups = [{'params': regular_params}, 
                        galore_param_dict]
        
        if optimizer_name=="galoradamw":
          optimizer = GaLoreAdamW(param_groups, **optimizer_kwargs)
        elif optimizer_name=="galoreadamw8bit":
          optimizer = GaLoreAdamW(param_groups, **optimizer_kwargs)


    if scheduler_name=="cosine_with_hard_restarts_schedule_with_warmup":

      scheduler=get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=total_update_steps*warmup_ratio,
                                                                num_training_steps=total_update_steps,
                                                                **scheduler_kwargs)
    elif scheduler_name=="cosine_schedule_with_warmup":

      scheduler=get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=total_update_steps*warmup_ratio,
                                                num_training_steps=total_update_steps,
                                                **scheduler_kwargs)


                                         
    return optimizer,scheduler


def load_and_prepare_dataset(dataset=None,dataset_dir:str=None,preprocess_func=None,fn_kwargs:dict=None):
  '''
  preprocess_func : define your own preprocessing function. This sholud take a dataset object as its argument
  '''

  if dataset is None and dataset_dir is None:
    raise "Either dataset or dataset_dir should be given"
  
  if dataset_dir is not None:
    try:
      dataset=load_dataset(dataset_dir)
      print("---load dataset from huggingface hub---")
    except:
      dataset=Dataset.load_from_disk(dataset_dir)
      print("---load dataset from local disk---")

  if preprocess_func is not None:
    dataset=dataset.map(preprocess_func,fn_kwargs=fn_kwargs)

  return dataset