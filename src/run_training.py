import torch
import numpy as np
import os, platform, warnings,sys
import argparse
import random
import ipdb
from prettytable import PrettyTable
import mlflow
from training_modules import *
from peft import LoraConfig
from preprocess_func import make_translation_input_from_dataset
from prompt_template import TranslationTemplate
from datetime import datetime

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
    parser.add_argument("--cache_dir",type=str)

    ## hyper parameters
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--optimizer",type=str,default="PagedAdam8bit",help='["Adam8bit", "AdamW", "PagedAdam8bit"]')
    parser.add_argument("--scheduler",type=str,default="cosine_with_hard_restarts_schedule_with_warmup",help='["cosine_with_hard_restarts_schedule_with_warmup", "cosine_schedule_with_warmup"]')
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--gradient_checkpointing",type=bool,default=False,help="reducing required memory size but slower training")
    parser.add_argument("--warmup_ratio",type=float,default=0.01)
    parser.add_argument("--num_save_per_epoch",type=int,default=3,help="number of saving(evaluating) per a epoch")
    
    ## lora config
    parser.add_argument("--lora",type=bool,help="train wtih lora, full finetuning otherwise",default=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout_rate", type=float, default=0.1)
    parser.add_argument("--lora_bias", default="none")
    parser.add_argument("--lora_task_type", type=str, default="CAUSAL_LM")
    parser.add_argument("--lora_target_modules", type=str, nargs='*', default=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","down_proj","up_proj"])

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

    return parser.parse_args()

def set_environ(args):
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["MLFLOW_EXPERIMENT_NAME"]=args.expr_name
    # os.environ["MLFLOW_RUN_ID"]

    if args.expr_desc is not None:
        os.environ["MLFLOW_TAGS"]='{"mlflow.note.content":' + f'"{args.expr_desc}"' + "}"

def main(args):
    ######################################### General Settings #########################################
    args.run_name=f"{args.run_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}"
    args.output_dir= f"{args.base_dir}/trained/{args.run_name}"

    try:
        local_rank=str(os.environ['LOCAL_RANK'])
    except KeyError: # single gpu
        local_rank="0"
    ####################################################################################################


    ######################################### model & tokenizer #########################################
    model=load_model(args.base_model_dir,gradient_checkpointing=args.gradient_checkpointing,quantization_config=None)
    model.config.use_cache = False # use_cache is only for infernce

    if model.config.max_position_embeddings<args.max_seq_length:
        model.config.max_position_embeddings=args.max_seq_length

    if args.lora:
      peft_config = LoraConfig(
              r=args.lora_rank,
              lora_alpha=args.lora_alpha,
              lora_dropout=args.lora_dropout_rate,
              bias=args.lora_bias,
              task_type=args.lora_task_type,
              target_modules=args.lora_target_modules
          )
    else:
      peft_config=None

    tokenizer=load_tokenizer(args.base_model_dir)
    if len(tokenizer)!=int(model.config.vocab_size):
        model.resize_token_embeddings(len(tokenizer))
    assert len(tokenizer)==int(model.config.vocab_size) , 'vocab sizes of the tokenizer and the model should be same'
    ######################################################################################################

    ######################################### dataset ####################################################
    train_dataset=load_and_prepare_dataset(dataset_dir=args.train_dataset_dir,preprocess_func=make_translation_input_from_dataset,fn_kwargs={"prompt_template":TranslationTemplate.prompt_template,"tokenizer":tokenizer})
    if args.eval:
        eval_dataset=load_and_prepare_dataset(args.eval_dataset_dir,preprocess_func=make_translation_input_from_dataset,fn_kwargs={"prompt_template":TranslationTemplate.prompt_template,"tokenizer":tokenizer})
    if local_rank=="0":
        for data in train_dataset.shuffle().select(range(5)):
            print("-------example-------\n",data[args.dataset_text_field])
    #######################################################################################################


    ######################################## Optimizer & Scheduler #########################################
    total_update_steps=int((len(train_dataset)*args.num_epochs)/(args.train_batch_size*args.gradient_accumulation_steps*torch.cuda.device_count()))
    optimizer,scheduler=load_optimizer_scheduler(model=model,
                                                optimizer_name=args.optimizer,
                                                scheduler_name=args.scheduler,
                                                total_update_steps=total_update_steps,
                                                warmup_ratio=args.warmup_ratio,
                                                optimizer_kwargs={"lr":args.learning_rate,"weight_decay":args.weight_decay},
                                                scheduler_kwargs={"num_cycles":0.3},
                                                )
    #######################################################################################################

    ######################################### Trainer Setiings #########################################
    if args.eval_steps is None:
        args.eval_steps=int(total_update_steps/args.num_save_per_epoch)

    create_trainer=CreateTrainer(args)
    training_arguments=create_trainer.training_arguments
    trainer=create_trainer.create_trainer_sft(model,tokenizer,optimizer,scheduler,train_dataset,eval_dataset=None,peft_config=peft_config,response_template=PromptTemplate.response_template)
    ######################################################################################################
    
    print("detected device : ",training_arguments.device)

    if local_rank=="0":
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

    # Logging dataset
    if local_rank=="0":
        last_run_id = mlflow.last_active_run().info.run_id
        with mlflow.start_run(run_id=last_run_id):
            mlflow.log_input(mlflow.data.from_huggingface(train_dataset,source=args.dataset_dir), context="training dataset")
            if args.eval:
                mlflow.log_input(mlflow.data.from_huggingface(eval_dataset,source=args.dataset_dir), context="evaluation dataset")

    
if __name__=="__main__":
  args=parse_args()
  seed_everything(args.seed)
  set_environ(args)
  main(args)
