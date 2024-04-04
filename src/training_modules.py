from transformers import (
  BitsAndBytesConfig,
  TrainingArguments,
    Trainer,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    AdamW,
    DefaultDataCollator,
  )
import bitsandbytes as bnb
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, platform, warnings,sys
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
import argparse
import random
from utils import *
import ipdb
from prettytable import PrettyTable
import mlflow
from training_modules import *
from translation_template import TranslationTemplate

class CreateTrainer():
  def __init__(self,args):
    self.args=args
    self._set_training_arguments()

  def _set_training_arguments(self):
    self.training_arguments = TrainingArguments(output_dir= args.output_dir,
        # fp16= True,
        bf16= True,
        # run_name=args.expr_desc,
        # metric_for_best_model="eval_loss",
       ddp_find_unused_parameters=False,
        # torch_compile=True,
                        )
    self.training_arguments=self.training_arguments.set_dataloader(train_batch_size=args.train_batch_size,
                                                             eval_batch_size=args.eval_batch_size,
                                                             pin_memory=True,
                                                             num_workers=args.num_workers,
                                                             sampler_seed=args.seed
                                                            )
    self.training_arguments=self.training_arguments.set_training(
        learning_rate= args.learning_rate,
        batch_size=args.train_batch_size,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        )
    self.training_arguments=self.training_arguments.set_logging(strategy="steps",steps=args.logging_steps,report_to=["mlflow"])
    self.training_arguments=self.training_arguments.set_save(strategy="steps",steps=args.eval_steps,total_limit=args.save_total_limit)

    if self.args.eval:
          training_arguments=training_arguments.set_evaluate(strategy="steps", batch_size=args.eval_batch_size,steps=args.eval_steps,delay=0)


  def create_trainer_sft(self,model,tokenizer,optimizer,scheduler,train_dataset,eval_dataset,peft_config=None,response_template=None,data_collator=None):

    if response_template is not None:
      response_template_with_context=f"\n{response_template}\n"
      response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
      data_collator=DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    else:
      if data_collator is None:
        data_collator=DefaultDataCollator()

    trainer = SFTTrainer(
    args=self.training_arguments,
    model=model,
    tokenizer=tokenizer,
    optimizers=(optimizer,scheduler),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if self.args.eval else None,
    data_collator=data_collator,
    peft_config=peft_config,
    max_seq_length= args.max_seq_length,
    dataset_text_field=args.dataset_text_field if args.dataset_text_field else None,
    )

    return trainer

  def create_trainer_basic(self):
    pass


def load_tokenizer(base_model_path,additional_special_tokens:list=None):
  """
  base_model_path : path or name of pretrained model
  """
  tokenizer = AutoTokenizer.from_pretrained(base_model_path,trust_remote_code=True)

  ###### set special tokens #####
  if additional_special_tokens is not None:
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

  if not tokenizer.pad_token or tokenizer.pad_token==tokenizer.eos_token:
    ## padding with eos_token might make repetiton in inference.
    tokenizer.pad_token=tokenizer.unk_token

  tokenizer.padding_side="right"
  return tokenizer


def load_model(base_model_path,
            gradient_checkpointing=False,
            quantization_config=None,
            flash_attn=True,
            cache_dir="/azurestorage/huggingface_cache/models",
            **kwargs,
            ):

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
                        ):

    params=filter(lambda x:x.requires_grad,model.parameters())
                  

    if optimizer_name=="AdamW":
        optimizer=AdamW(params=params,
                        **optimizer_kwargs,
                        )

    elif optimizer_name=="Adam8bit":
        optimizer = bnb.optim.Adam8bit(params=params,
                        **optimizer_kwargs,
                        )

        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )

    elif optimizer_name=="PagedAdam8bit":
        optimizer = bnb.optim.PagedAdam8bit(params=params,
                        **optimizer_kwargs,
                        )

        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )

    if scheduler_name=="cosine_with_hard_restarts_schedule_with_warmup"

      scheduler=get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                  num_warmup_steps=total_update_steps*warmup_ratio,
                                                                  num_training_steps=total_update_steps,
                                                                  **scheduler_kwargs)
    elif scheduler_name=="cosine_schedule_with_warmup"

      scheduler=get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                  num_warmup_steps=total_update_steps*warmup_ratio,
                                                                  num_training_steps=total_update_steps,
                                                                  **scheduler_kwargs)
                                         
    return optimizer,scheduler


def load_and_prepare_dataset(dataset=None,dataset_dir:str=None,preprocess_func=None,fn_kwargs:dict=None):
  '''
  preprocess_func : define your own preprocessing function. This sholud take dataset object as its argument
  '''

  if dataset is None and dataset_dir is None:
    raise "Either dataset or dataset_dir should be given"
  
  if dataset_dir is not None:
    try:
      dataset=load_datasets(dataset_dir)
      print("load dataset from huggingface hub")
    except:
      dataset=Dataset.load_from_disk(dataset_dir)
      print("load dataset from local disk")

  if preprocess_func is not None:
    dataset=dataset.map(preprocess_func,fn_kwargs=fn_kwargs)

    return dataset