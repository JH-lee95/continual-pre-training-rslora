from transformers import (
  AutoModelForCausalLM, 
  AutoTokenizer, 
  BitsAndBytesConfig,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    AdamW,
  )
import bitsandbytes as bnb
import os, warnings,sys
from datasets import load_dataset,Dataset,concatenate_datasets
import torch
from utils import *
import ipdb
import torch.nn as nn


def load_tokenizer(base_model_path,additional_special_tokens:list=None):
  """
  base_model_path : path or name of pretrained model
  chat_template : jinja chat template
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
            cache_dir="/root/azurestorage/huggingface_cache/models"):

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        trust_remote_code=True, 
        use_cache=False if gradient_checkpointing else True, # use_cache is incompatible with gradient_checkpointing
        torch_dtype="auto",
        # device_map="cuda",
        attn_implementation="flash_attention_2" if flash_attn else None,
        cache_dir="/root/azurestorage/huggingface_cache/models",
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def load_and_prepare_dataset(tokenizer,         
                            seed,
                            max_len,
                            translation_template_wo_term_dict:str=None,
                            translation_template_w_term_dict:str=None,
                            glossary_template:str=None,
                            glossary_tags:str=None,):

    dataset=Dataset.load_from_disk("/nvme_temp/prepared_for_training/training_dataset_20k")
    dataset=dataset.map(make_translation_input_from_dataset,
                        fn_kwargs={
                            "tokenizer":tokenizer,
                            "translation_template_wo_term_dict":translation_template_wo_term_dict,
                            "translation_template_w_term_dict":translation_template_w_term_dict,
                            "glossary_template":glossary_template,
                            "glossary_tags":glossary_tags,
                              }
                                )

    return dataset


def load_and_prepare_dataset_cpo(tokenizer,         
                            seed,
                            max_len,
                            translation_template_wo_term_dict:str=None,
                            translation_template_w_term_dict:str=None,
                            glossary_template:str=None,
                            glossary_tags:str=None,):

    dataset=Dataset.load_from_disk("/nvme_temp/prepared_for_training/cpo_dataset_10k_eval_by_gemba")
    dataset=dataset.map(make_translation_input_from_dataset,
                        fn_kwargs={
                            "tokenizer":tokenizer,
                            "translation_template_wo_term_dict":template_wo_term_dict,
                            "translation_template_w_term_dict":template_w_term_dict,
                            "glossary_template":glossary_template,
                            "glossary_tags":glossary_tags,
                            "output":False,
                              }
                                )

    dataset=dataset.rename_columns({"text":"prompt"})
    dataset=dataset.select_columns(["prompt","chosen","reject"])

    return dataset


    return dataset


def load_optimizer_scheduler(model,
                        learning_rate, 
                        weight_decay,
                        total_update_steps, 
                        warmup_ratio,   
                        quantize=True,  
                        **kwargs
                        ):

    if quantize:
        # optimizer = bnb.optim.Adam8bit(params=filter(lambda x:x.requires_grad,model.parameters()), 
        #                             lr=learning_rate, 
        #                             weight_decay=weight_decay,
        #                             )

        optimizer= bnb.optim.PagedAdam8bit(params=filter(lambda x:x.requires_grad,model.parameters()), 
                                    lr=learning_rate, 
                                    weight_decay=weight_decay,
                                    )

        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )         
    else:
        optimizer=AdamW(params=filter(lambda x:x.requires_grad,model.parameters()),
                        lr=learning_rate,
                        weight_decay=weight_decay
                        )


    scheduler=get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=total_update_steps*warmup_ratio,
                                                                num_cycles=0.3,
                                                                num_training_steps=total_update_steps)

    # scheduler=get_cosine_schedule_with_warmup(optimizer,
    #                                         num_warmup_steps=total_update_steps*warmup_ratio,
    #                                         num_cycles=0.3,
    #                                         num_training_steps=total_update_steps)
    return optimizer,scheduler
