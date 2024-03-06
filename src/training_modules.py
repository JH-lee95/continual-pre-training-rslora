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


def load_and_prepare_dataset(tokenizer,seed,max_len,metric=True):

    response_template_with_context = "\n### Output:\n"
 
    columns=["korean","english","src","tgt"]
  
    # dataset=prepare_translation_dataset("/root/azurestorage/data/번역데이터셋/aligned_dataset/final_dataset/","/root/azurestorage/data/번역데이터셋/aligned_dataset/term_dict_result_dedup.jsonl")    # dataset=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/final_dataset_with_term_dict")

    dataset=Dataset.load_from_disk("/nvme_temp/prepared_for_training/training_dataset_20k")
    dataset=dataset.map(make_translation_prompt,fn_kwargs={"tokenizer":tokenizer})

    # dataset=dataset.filter(lambda x:len(tokenizer.tokenize(x["text"]))<max_len) # to guarantee perfect completion up to eos token,

    # eval_dataset_1=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/prepared_for_training/translation_dataset_valid_wo_term_dict").select(range(10))
    # eval_dataset_2=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/prepared_for_training/translation_dataset_valid_w_term_dict").select(range(10))
    # eval_dataset_3=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/prepared_for_training/flores_ko_eng/dev").select(range(10))

    # eval_dataset=concatenate_datasets([eval_dataset_1.select_columns(columns),eval_dataset_2.select_columns(columns),eval_dataset_3.select_columns(columns)])
    # eval_dataset=eval_dataset.map(make_translation_prompt,fn_kwargs={"tokenizer":tokenizer})

    # if metric:
    #     metric=MetricCollection(len(eval_dataset_1),len(eval_dataset_2),tokenizer=tokenizer,response_template=response_template_with_context)

    #     return dataset,eval_dataset,metric

    # else:
        # return dataset,eval_dataset

    return dataset


def load_and_prepare_dataset_cpo(tokenizer,seed,max_len,metric=True):

    response_template_with_context = "\n### Translation:\n"
 
    columns=["korean","english","src","tgt"]
  
    # dataset=prepare_translation_dataset("/root/azurestorage/data/번역데이터셋/aligned_dataset/final_dataset/","/root/azurestorage/data/번역데이터셋/aligned_dataset/term_dict_result_dedup.jsonl")    # dataset=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/final_dataset_with_term_dict")
    dataset=Dataset.load_from_disk("/nvme_temp/prepared_for_training/cpo_9k")
    # dataset=dataset.map(make_translation_prompt,fn_kwargs={"tokenizer":tokenizer})
    # dataset=dataset.filter(lambda x:len(tokenizer.tokenize(x["text"]))<max_len) # to guarantee perfect completion up to eos token,


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

    return optimizer,scheduler



class MetricCollection:
    def __init__(self, dataset_1_size,dataset_2_size,tokenizer=None,response_template=None,ignore_index=-100):
        self.ignore_index=ignore_index
        self.dataset_1_size = dataset_1_size # w/o term_dict
        self.dataset_2_size=  dataset_2_size # w term_dict
        self.epsilon: float = 0.1
        self.tokenizer=tokenizer
        # self.response_token_ids= self.tokenizer.encode(response_template, add_special_tokens=False)[2:]
        

    def compute_loss(self, model_output, labels, shift_labels=True):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output
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


    # def compute_loss(self, model_output, labels, shift_labels=True):
    #     logits = model_output["logits"] if isinstance(model_output, dict) else model_output
    #     if shift_labels:
    #         logits = logits[..., :-1, :]  # Contiguous not needed for NumPy
    #         labels = labels[..., 1:]

    #     log_probs = -np.log(np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True))  # NumPy log_softmax
    #     if labels.ndim == log_probs.ndim - 1:
    #         labels = labels[..., np.newaxis]  # NumPy equivalent of unsqueeze

    #     padding_mask = labels == self.ignore_index
    #     labels = np.clip(labels, a_min=0, a_max=None)  # NumPy clamp
    #     nll_loss = log_probs[np.arange(len(log_probs)), labels]  # NumPy gather
    #     smoothed_loss = np.log(log_probs.sum(axis=-1, keepdims=True) + 1e-8)  # Add smoothing for numerical stability

    #     nll_loss[padding_mask] = 0.0
    #     smoothed_loss[padding_mask] = 0.0

    #     num_active_elements = np.prod(padding_mask.shape) - np.sum(padding_mask)
    #     nll_loss = np.mean(nll_loss)  # Direct mean calculation
    #     smoothed_loss = np.mean(smoothed_loss)  # Direct mean calculation
    #     return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


    # def mask_labels(self,labels):
    #     # mask labels except generation candidates

    #     for i in range(len(labels)):
    #         response_token_ids_start_idx = None

    #         for idx in np.where(labels[i] == self.response_token_ids[0])[0]:
    #             # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
    #             if (
    #                 self.response_token_ids
    #                 == labels[i][idx : idx + len(self.response_token_ids)].tolist()
    #             ):
    #                 response_token_ids_start_idx = idx

    #         if response_token_ids_start_idx is None:
    #             warnings.warn(
    #                 f"Could not find response key `{self.response_template}` in the "
    #                 f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
    #                 f"This instance will be ignored in loss calculation. "
    #                 f"Note, if this happens often, consider increasing the `max_seq_length`."
    #             )
    #             labels[i, :] = self.ignore_index
    #         else:
    #             response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

    #             # Make pytorch loss function ignore all tokens up through the end of the response key
    #             labels[i, :response_token_ids_end_idx] = self.ignore_index

    #     return labels

    def compute_metrics(self, p):

        metrics = {}

        # response_template_ids=torch.tensor(self.response_template_ids).expand(p.label_ids.shape[0],-1)
        # p.label_ids=self.mask_labels(p.label_ids)

        labels_1 = p.label_ids[:self.dataset_1_size]
        labels_2 = p.label_ids[self.dataset_1_size:self.dataset_1_size+self.dataset_2_size]
        labels_3 = p.label_ids[self.dataset_1_size+self.dataset_2_size:]

        predictions_1 = p.predictions[:self.dataset_1_size, :]
        predictions_2 = p.predictions[self.dataset_1_size:self.dataset_1_size+self.dataset_2_size, :]
        predictions_3 = p.predictions[self.dataset_1_size+self.dataset_2_size:, :]

        metrics['eval_loss_w/o_term_dict'] = self.compute_loss(torch.tensor(predictions_1),torch.tensor(labels_1)).item()
        metrics['eval_loss_w_term_dict'] = self.compute_loss(torch.tensor(predictions_2),torch.tensor(labels_2)).item()
        metrics['eval_loss_flores'] = self.compute_loss(torch.tensor(predictions_3),torch.tensor(labels_3)).item()

        # ipdb.set_trace()

        return metrics