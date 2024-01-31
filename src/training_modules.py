from transformers import (
  AutoModelForCausalLM, 
  AutoTokenizer, 
  BitsAndBytesConfig,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    AdamW,
  )
import bitsandbytes as bnb
import os, warnings,sys
from datasets import load_dataset,Dataset,concatenate_datasets
import torch
from utils import *


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
        # device_map="cuda",
        attn_implementation="flash_attention_2",
        cache_dir="/root/azurestorage/huggingface_cache/models",
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def load_and_prepare_dataset(tokenizer,seed,max_len):
 
  
    # dataset=prepare_translation_dataset("/root/azurestorage/data/번역데이터셋/aligned_dataset/final_dataset/","/root/azurestorage/data/번역데이터셋/aligned_dataset/term_dict_result_dedup.jsonl")    # dataset=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/final_dataset_with_term_dict")
    dataset=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/prepared_for_training/translation_dataset_training")
    dataset=dataset.map(make_translation_prompt,fn_kwargs={"tokenizer":tokenizer})
    dataset=dataset.filter(lambda x:len(tokenizer.tokenize(x["text"]))<max_len) # to guarantee perfect completion up to eos token,

    # eval_dataset=load_dataset("jhflow/flores_ko_eng",token="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",split="dev")
    eval_dataset=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/prepared_for_training/translation_dataset_valid")
    eval_dataset=eval_dataset.map(make_translation_prompt,fn_kwargs={"tokenizer":tokenizer})
    return dataset,eval_dataset


def load_optimizer_scheduler(model,
                        learning_rate, 
                        weight_decay,
                        total_update_steps, 
                        warmup_ratio,   
                        quantize=True,  
                        **kwargs
                        ):

    if quantize:
        optimizer = bnb.optim.Adam8bit(params=filter(lambda x:x.requires_grad,model.parameters()), 
                                    lr=learning_rate, 
                                    weight_decay=weight_decay,
                                    )

        # optimizer= bnb.optim.PagedAdam8bit(params=filter(lambda x:x.requires_grad,model.parameters()), 
        #                             lr=learning_rate, 
        #                             weight_decay=weight_decay,
        #                             )

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
                                                                num_cycles=5,
                                                                num_training_steps=total_update_steps)

    return optimizer,scheduler



class MetricCollection:
    def __init__(self, dataset_1_size,dataset_2_size,ignore_index=-100):
        self.ignore_index=ignore_index
        self.dataset_1_size = dataset_1_size
        self.dataset_2_size=  dataset_2_size

    def mymetric(labels, predicted_scores):
        ...
        return result

    def compute_loss(self, model_output, labels, shift_labels=True):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

    def compute_metrics(self, p: EvalPrediction) -> Dict:
        metrics = {}

        labels_1 = p.label_ids[:self.dataset_1_size]
        labels_2 = p.label_ids[self.dataset_1_size:self.dataset_2_size]
        labels_3 = p.label_ids[self.dataset_2_size]

        predictions_1 = p.predictions[:self.dataset_1_size, :]
        predictions_2 = p.predictions[self.dataset_1_size:self.dataset_2_size, :]
        predictions_3 = p.predictions[self.dataset_2_size:, :]

        metrics['eval_loss_w/o_term_dict'] = compute_loss(predictions_1,labels_1)
        metrics['eval_loss_w_term_dict'] = compute_loss( predictions_2,labels_2)
        metrics['eval_loss_flores'] = compute_loss(predictions_3,labels_3)

        return metrics