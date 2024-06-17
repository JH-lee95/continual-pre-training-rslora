import torch
import numpy as np
import os, platform, warnings,sys
import argparse
import random
import ipdb
from prettytable import PrettyTable
import mlflow
from training_modules import *
from datetime import datetime
from datasets import concatenate_datasets,load_dataset,Dataset,DatasetDict
import datasets

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def parse_args():
    parser = argparse.ArgumentParser()

    ## directories
    parser.add_argument("--base_dir",type=str,default="/azurestorage/models",help="base directory to save logs and checkpoints")
    parser.add_argument("--base_model_dir",type=str,required=True,help="local or huggingface hub directory of base model")
    parser.add_argument("--ckpt_dir",type=str,default=None)
    parser.add_argument("--train_dataset_dir",type=str)
    parser.add_argument("--eval_dataset_dir",type=str,default=None)
    parser.add_argument("--cache_dir",type=str,default=None)

    ## hyper parameters
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--optimizer",type=str,default="AdamW")
    parser.add_argument("--scheduler",type=str,default="cosine_schedule_with_warmup")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=4) 
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--gradient_checkpointing",action="store_true",help="reducing required memory size but slower training")
    parser.add_argument("--warmup_ratio",type=float,default=0.01)
    parser.add_argument("--num_save_per_epoch",type=int,default=3,help="number of saving(evaluating) per a epoch")
    
    ## lora config
    parser.add_argument("--enable_lora",action="store_true",help="train wtih lora, full finetuning otherwise")
    parser.add_argument("--lora_rank", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout_rate", type=float, default=0.01)
    parser.add_argument("--lora_bias", default="none")
    parser.add_argument("--lora_task_type", type=str, default="CAUSAL_LM")
    parser.add_argument("--use_rslora", type=bool, default=True)
    parser.add_argument("--lora_target_modules", type=str, nargs='*', default=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","down_proj","up_proj","embed_tokens", "lm_head"])
    
    ## etc
    parser.add_argument("--logging_steps",type=int,default=100)
    parser.add_argument("--eval_steps",type=int,default=None)
    parser.add_argument("--save_total_limit",type=int,default=10)
    parser.add_argument("--expr_name",type=str,default=None, help="experiment name",required=True)
    parser.add_argument("--expr_desc",type=str,help = "description for experiment", default = None)
    parser.add_argument("--run_name",type=str,help = "run name", default = None)
    parser.add_argument("--train",type=bool, default=True)

    parser.add_argument("--eval",type=bool, default=False)
    parser.add_argument("--test",type=bool, default=False)
    parser.add_argument("--dataset_text_field",type=str, default="text")
    parser.add_argument("--use_unsloth",action="store_true",help="use usloth backend for training")
    parser.add_argument("--device_map",type=str,help = "device map to load the model ex)'auto' for model parallel or'cuda", default = None)


    return parser.parse_args()

def set_environ(args):
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["MLFLOW_EXPERIMENT_NAME"]=args.expr_name
    # os.environ["MLFLOW_RUN_ID"]

    if args.expr_desc is not None:
        os.environ["MLFLOW_TAGS"]='{"mlflow.note.content":' + f'"{args.expr_desc}"' + "}"


def print_training_information(args,model,dataset,training_arguments,total_update_steps):

    if os.getenv("LOCAL_RANK","0")=="0":

        print("---Training data samples---")
        for data in dataset.shuffle().select(range(10)):
            print("--------------------\n",data[args.dataset_text_field])

        args_table = PrettyTable(["Argument", "Value"])
        for k,v in training_arguments.__dict__.items():
            args_table.add_row([k,v])
        args_table.add_row(["total_update_steps",total_update_steps])
        print(args_table)
        model.print_trainable_parameters()


def main(args):
    ######################################### General Settings #########################################
    args.run_name=f"{args.run_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}"
    args.output_dir= f"{args.base_dir}/trained/{args.run_name}"
    ####################################################################################################

    ######################################### model & tokenizer #########################################
    model,tokenizer=load_model_tokenizer(base_model_path=args.base_model_dir,gradient_checkpointing=args.gradient_checkpointing,cache_dir=args.cache_dir,use_unsloth=args.use_unsloth,pad_token=None,device_map=args.device_map)
 
    if model.config.max_position_embeddings<args.max_seq_length:
        model.config.max_position_embeddings=args.max_seq_length

    tokenizer.bos_token="<|im_start|>"
    tokenizer.eos_token="<|im_end|>"
    tokenizer.pad_token="<|endoftext|>"


    if args.enable_lora:
        model=get_lora_model(model,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout_rate,
                bias=args.lora_bias,
                task_type=args.lora_task_type,
                target_modules=args.lora_target_modules,
                use_rslora=args.use_rslora
                ) # rank stabilized lora

    ######################################################################################################

    ######################################### dataset ####################################################
    dataset_ko=DatasetDict.load_from_disk("/nvme0/data/KOREAN-WEBTEXT-sampled_150K_380MT")
    dataset_en=DatasetDict.load_from_disk("/nvme0/data/fineweb-edu-sampled_15K_38MT")

    train_dataset=concatenate_datasets([dataset_ko["train"].select_columns("text"),dataset_en["train"].select_columns("text")]).shuffle().select(range(10000))
    eval_dataset=concatenate_datasets([dataset_en["test"].select_columns("text"),dataset_en["test"].select_columns("text")]).shuffle().select(range(1000))

    train_dataset=train_dataset.map(lambda x: {"text":f"{tokenizer.bos_token}{x['text']}{tokenizer.eos_token}"},num_proc=16)
    eval_dataset=eval_dataset.map(lambda x: {"text":f"{tokenizer.bos_token}{x['text']}{tokenizer.eos_token}"},num_proc=16)
    #######################################################################################################


    ######################################## Optimizer & Scheduler #########################################
    total_update_steps=int((len(train_dataset)*args.num_epochs)/(args.train_batch_size*args.gradient_accumulation_steps*torch.cuda.device_count()))
    optimizer_kwargs={"weight_decay":args.weight_decay}
    scheduler_kwargs={"num_cycles":0.3}

    optimizer,scheduler=load_optimizer_scheduler(model,args.optimizer,args.scheduler,total_update_steps,args.warmup_ratio,args.learning_rate,optimizer_kwargs,scheduler_kwargs)
    #######################################################################################################

    ######################################### Trainer Settings #########################################
    if args.eval_steps is None:
        args.eval_steps=int(total_update_steps/args.num_save_per_epoch)

    # args.eval_steps=2

    create_trainer=CreateTrainer(args)
    training_arguments=create_trainer.training_arguments
    trainer=create_trainer.create_trainer_sft(model,tokenizer,optimizer,scheduler,train_dataset,eval_dataset=eval_dataset)
    print("detected device : ",training_arguments.device)
    ######################################################################################################
    
    print_training_information(args,model,train_dataset,training_arguments,total_update_steps)

    if args.train:
      if args.ckpt_dir is not None:
        trainer.train(args.ckpt_dir)
      else:
        trainer.train()

    # # Logging dataset
    # if os.getenv("LOCAL_RANK","0"):
    #     last_run_id = mlflow.last_active_run().info.run_id
    #     with mlflow.start_run(run_id=last_run_id):
    #         mlflow.log_input(mlflow.data.from_huggingface(train_dataset,data_files=args.train_dataset_dir), context="training dataset")
    #         if args.eval:
    #             mlflow.log_input(mlflow.data.from_huggingface(eval_dataset,data_files=args.eval_dataset_dir), context="evaluation dataset")

    
if __name__=="__main__":
  args=parse_args()
  seed_everything(args.seed)
  set_environ(args)
  main(args)