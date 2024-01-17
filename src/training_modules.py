from transformers import (
  AutoModelForCausalLM, 
  AutoTokenizer, 
  BitsAndBytesConfig,
  TrainingArguments,
    Trainer,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    AdamW,
  )
import bitsandbytes as bnb
import os, platform, warnings,sys
from datasets import load_dataset,Dataset,concatenate_datasets
import torch
from utils import make_translation_prompt,add_src_tgt_tag


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
            cache_dir="/root/azurestorage/huggingface_cache/models"):

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        trust_remote_code=True, 
        use_cache=False if gradient_checkpointing else True, # use_cache is incompatible with gradient_checkpointing
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        cache_dir="/root/azurestorage/huggingface_cache/models",
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def load_and_prepare_dataset(tokenizer,seed):
 
    dataset_group0=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/dedup/group0")
    dataset_group1=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/dedup/group1")
    dataset_group2=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/dedup/group2")
    dataset_group3=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/dedup/group3")
    dataset=concatenate_datasets([dataset_group0,dataset_group1,dataset_group2,dataset_group3]).shuffle(seed=seed)
    # dataset=prepare_translation_dataset("/root/azurestorage/data/번역데이터셋/aligned_dataset/translation_sampled_120k/","/root/azurestorage/data/번역데이터셋/aligned_dataset/term_dict_result.jsonl")
    
    dataset=dataset.
    dataset=dataset.map(make_translation_prompt,fn_kwargs={"tokenizer":tokenizer})
    dataset=dataset.filter(lambda x:len(tokenizer.tokenize(x["text"]))<args.max_len) # to guarantee perfect completion up to eos token,
    print("-------example-------\n",dataset[0]["text"])

    flores_eval_dataset=load_dataset("jhflow/flores_ko_eng",token="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")
    flores_eval_dataset=flores_eval_dataset.map(make_translation_prompt,fn_kwargs)
    
    return dataset,flores_eval_dataset


def load_optimizer_scheduler(model,
                        learning_rate, 
                        weight_deacy,
                        total_update_steps, 
                        warmup_ratio,   
                        quantize=True,  
                        **kwargs
                        ):

    if quantize:
        optimizer = bnb.optim.Adam8bit(params=filter(lambda x:x.requires_grad,model.parameters()), 
                                    lr=args.learning_rate, 
                                    weight_deacy=args.weight_decay,
                                    )

        # optimizer= bnb.optim.PagedAdam8bit(params=filter(lambda x:x.requires_grad,model.parameters()), 
        #                             lr=args.learning_rate, 
        #                             weight_deacy=args.weight_decay,
        #                             )

        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )         
    else:
        optimizer=AdamW(params=filter(lambda x:x.requires_grad,model.parameters()),
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay
                        )


    scheduler=get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=total_update_steps*warmup_ratio,
                                                                num_cycles=5,
                                                                num_training_steps=total_update_steps)

    return optimizer,scheduler

