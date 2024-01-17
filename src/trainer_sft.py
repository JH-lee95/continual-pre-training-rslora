from transformers import (
  BitsAndBytesConfig,
  TrainingArguments,
    Trainer,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    AdamW,
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

def parse_args():

    parser = argparse.ArgumentParser()

    ## directories
    parser.add_argument("--base_dir",type=str,default="/root/azurestorage/models",help="base directory to save logs and checkpoints")
    parser.add_argument("--base_model_dir",type=str,required=True,help="local or huggingface hub directory of base model")
    parser.add_argument("--ckpt_dir",type=str,default=None)
    parser.add_argument("--mlflow_dir",type=str, default="mlruns")
    parser.add_argument("--dataset_dir",type=str)

    ## hyper parameters
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--learning_rate",type=float, default=5e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch_size', type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--gradient_checkpointing",type=bool,default=False,help="reducing required memory size but slower training")
    parser.add_argument("--warmup_ratio",type=float,default=0.01)
    parser.add_argument("--full_ft",action="store_true",help="full finetuning otherwise lora")
    parser.add_argument("--num_save_per_epoch",type=int,default=3,help="number of saving(evaluating) per a epoch")
    
    ## etc
    parser.add_argument("--expr_name",type=str,default=None, help="experiment name",required=True)
    parser.add_argument("--expr_desc",type=str,help = "description for experiment", default = None)
    parser.add_argument("--train",type=bool, default=True)
    parser.add_argument("--test",type=bool, default=False)
    parser.add_argument("--local_rank")

    return parser.parse_args()

def set_environ(args):
    os.environ["MLFLOW_EXPERIMENT_NAME"]=args.expr_name
    # os.environ["MLFLOW_TRACKING_URI"]=args.mlflow_dir
    # os.environ["HF_MLFLOW_LOG_ARTIFACTS"]="1"
    os.environ["MLFLOW_TAGS"]='{"mlflow.note.content":' + f'"{args.expr_desc}"' + "}"


def main(args):

    ######################################### model #########################################
    model=load_model(args.base_model_dir,gradient_checkpointing=args.gradient_checkpointing,quantization_config=None)
    model=model.to("cuda")
    model.config.use_cache = False # use_cache is only for infernce

    if model.config.max_position_embeddings<args.max_len:
        model.config.max_position_embeddings=args.max_len

    # model.config.pretraining_tp = 1 
    
    if not args.full_ft:
      ## peft (lora)
      peft_config = LoraConfig(
              r=32,
              lora_alpha=32,
              lora_dropout=args.dropout_rate,
              bias="none",
              task_type="CAUSAL_LM",
              target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","down_proj","up_proj"]
          )
      
      model=get_peft_model(model,peft_config)
    #######################################################################################


    ######################################### tokenizer & dataset #########################################
    tokenizer=load_tokenizer(args.base_model_dir)

    print(f"eos_token is : {tokenizer.eos_token}")
    
    train_dataset,eval_dataset=load_and_prepare_dataset(tokenizer=tokenizer,seed=args.seed)
    # train_dataset=load_and_prepare_dataset(tokenizer)
    
    if len(tokenizer)!=int(model.config.vocab_size):
        model.resize_token_embeddings(len(tokenizer))
    assert len(tokenizer)==int(model.config.vocab_size) , 'vocab sizes of the tokenizer and the model should be same'
    print(f"len tokenizer : {len(tokenizer)}")
    print(f"model token size : {model.config.vocab_size}")
    #######################################################################################################
    
    
    ######################################## Optimizer & Scheduler #########################################
    total_update_steps=int((len(train_dataset)*args.epoch_size)/(args.batch_size*args.gradient_accumulation_steps))

    optimizer,scheduler=load_optimizer_scheduler(model=model,
                                                learning_rate=args.learning_rate,
                                                weight_decay=args.weight_decay,
                                                total_update_steps=total_update_steps,
                                                warmup_ratio=args.warmup_ratio,
                                                quantize=True)
    #######################################################################################################

    ######################################### Trainer Setiings #########################################
    expr_name=f"{args.expr_name}_v1"
    output_dir= f"{args.base_dir}/trained/{expr_name}"

    while os.path.exists(output_dir):
        expr_name=expr_name.split("_v")[0]+"_v"+str(int(expr_name.split("_v")[-1])+1)
        output_dir= f"{args.base_dir}/trained/{expr_name}"
        

    eval_steps=int(total_update_steps/args.num_save_per_epoch)
    # eval_steps=100

    training_arguments = TrainingArguments(output_dir= output_dir,
        # fp16= True,
        bf16= True,
        # run_name=args.expr_desc,
        metric_for_best_model="eval_loss",
       # ddp_find_unused_parameters=False,
        # torch_compile=True,
                        )
    training_arguments=training_arguments.set_dataloader(train_batch_size=args.batch_size,
                                                             eval_batch_size=args.batch_size*2,
                                                             pin_memory=True,
                                                             num_workers=4,
                                                             sampler_seed=args.seed)
    training_arguments=training_arguments.set_training(
        learning_rate= args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_epochs=args.epoch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        )
    training_arguments=training_arguments.set_logging(strategy="steps",steps=100,report_to=["mlflow"])
    training_arguments=training_arguments.set_evaluate(strategy="steps", batch_size=args.batch_size*2,steps=eval_steps,delay=0)
    training_arguments=training_arguments.set_save(strategy="steps",steps=eval_steps,total_limit=20,)
    training_arguments.load_best_model_at_end=True

    response_template_with_context = "\n### Output:\n"  # We added context here: "\n". This is enough for this tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    collator=DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    trainer = SFTTrainer(
    args=training_arguments,
    model=model,
    tokenizer=tokenizer,
    optimizers=(optimizer,scheduler),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # data_collator=collator,
    peft_config=peft_config if not args.full_ft else None,
    max_seq_length= args.max_len,
    dataset_text_field="text",
    # packing= True,
    )
    ######################################################################################################

    args_table = PrettyTable(["Argument", "Value"])
    for k,v in training_arguments.__dict__.items():
        args_table.add_row([k,v])
    args_table.add_row(["total_update_steps",total_update_steps])
    print(args_table)

    if args.train:
      if args.ckpt_dir is not None:
        trainer.train(args.ckpt_dir)
      else:
        trainer.train()
        trainer.save_model(output_dir+"/full_model")    
        
    mlflow.end_run()
    
if __name__=="__main__":
  args=parse_args()
  seed_everything(args.seed)
  set_environ(args)
  main(args)
