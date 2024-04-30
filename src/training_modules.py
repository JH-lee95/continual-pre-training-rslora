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
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit


class CreateTrainer():
  def __init__(self,args):
    self.args=args
    self._set_training_arguments()

  def _set_training_arguments(self):

    self.training_arguments = TrainingArguments(
      output_dir= self.args.output_dir,
        # fp16= True,
        bf16= True,
        run_name=self.args.run_name,
        lr_scheduler_kwargs=self.args.scheduler_kwargs,
        optim_target_modules=[r'*attn*',r'*mlp*'] if self.args.enable_galore else None
        ddp_find_unused_parameters=False,
                        )


    self.training_arguments=self.training_arguments.set_optimizer(name=self.args.optimizer.lower(),
                                                                  learning_rate=self.args.learning_rate,
                                                                  weight_decay=self.args.weight_decay,
                                                                  optim_args=self.args.optim_args,
                                                                  )
    self.training_arguments=self.training_arguments.set_lr_scheduler(name=self.args.scheduler.lower(),
                                                                    num_epochs=self.args.num_epochs,
                                                                    warmup_ratio=self.args.warmup_ratio)
    self.training_arguments=self.training_arguments.set_dataloader(train_batch_size=self.args.train_batch_size,
                                                             eval_batch_size=self.args.eval_batch_size,
                                                             pin_memory=True,
                                                             num_workers=self.args.num_workers,
                                                             sampler_seed=self.args.seed
                                                            )
    self.training_arguments=self.training_arguments.set_training(
        num_epochs=self.args.num_epochs,
        gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        seed=self.args.seed,
        )
    self.training_arguments=self.training_arguments.set_logging(strategy="steps",steps=self.args.logging_steps,report_to=["mlflow"])
    self.training_arguments=self.training_arguments.set_save(strategy="steps",steps=self.args.eval_steps,total_limit=self.args.save_total_limit)

    if self.args.eval:
          training_arguments=training_arguments.set_evaluate(strategy="steps", batch_size=self.args.eval_batch_size,steps=self.args.eval_steps,delay=0)


  def create_trainer_sft(self,model,tokenizer,train_dataset,eval_dataset,response_template=None,data_collator=None):

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
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if self.args.eval else None,
    data_collator=data_collator,
    max_seq_length= self.args.max_seq_length,
    dataset_text_field=self.args.dataset_text_field if self.args.dataset_text_field else None,
    )


    return trainer

  def create_trainer_basic(self):
    pass

def load_model_tokenizer(base_model_path,
            task_type:str="causal_lm",
            gradient_checkpointing=False,
            flash_attn=True,
            cache_dir="/azurestorage/huggingface_cache/models",
          additional_special_tokens=None,
          pad_token=None,
          pad_token_id=None,
          **model_kwargs,
          **tokenizer_kwargs,
            ):

  ##################### set model #######################
  if task_type.lower()=="causal_lm":

    if self.args.use_unsloth:
      model, _ = FastLanguageModel.from_pretrained(
                        model_name = base_model_path # Supports Llama, Mistral - replace this!
                        dtype=None,
                        # cache_dir=cache_dir,
                        **model_kwargs,
                    )

    else:
      model = AutoModelForCausalLM.from_pretrained(
          base_model_path,
          trust_remote_code=True,
          use_cache=False if gradient_checkpointing else True, # use_cache is incompatible with gradient_checkpointing
          torch_dtype="auto",
          # device_map="cuda",
          attn_implementation="flash_attention_2" if flash_attn else None,
          cache_dir=cache_dir,
          **model_kwargs,
      )

  elif task_type.lower()=="sequence_classification":
    pass

  elif task_type.lower()=="question_answering":
    pass

  if gradient_checkpointing:
      model.gradient_checkpointing_enable()

  ##################### set tokenizer #######################
  tokenizer = AutoTokenizer.from_pretrained(base_model_path,trust_remote_code=True,**tokenizer_kwargs)
  ###### set special tokens #####
  if additional_special_tokens is not None:
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

  if not tokenizer.pad_token or tokenizer.pad_token==tokenizer.eos_token:
    ## padding with eos_token might make repetiton in inference.

    if tokenizer.unk_token:
      tokenizer.pad_token=tokenizer.unk_token
    if pad_token is not None:
      tokenizer.pad_token=pad_token
    if pad_token_id is not None:
      tokenizer.pad_token_id=pad_token_id

  tokenizer.padding_side="right"

  if len(tokenizer)!=int(model.config.vocab_size):
      model.resize_token_embeddings(len(tokenizer))
  assert len(tokenizer)==int(model.config.vocab_size) , 'vocab sizes of the tokenizer and the model should be same'


  return model,tokenizer

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


def get_lora_model(model,use_unsloth=False,**lora_kwargs):

  if use_unsloth:
    model=FastLanguageModel.get_peft_model(model,**lora_kwargs)

  else:
    lora_config=LoraConfig(**lora_kwargs)
    model=get_peft_model(model,lora_config)

  if os.getenv('LOCAL_RANK',"0")=="0":
      model.print_trainable_parameters()

  return model