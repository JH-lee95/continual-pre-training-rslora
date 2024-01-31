import os,sys
import random
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset,Dataset,concatenate_datasets
import jsonlines
import ipdb
import json

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def check_if_translation_orca(sample):

    for d in sample:
        if "translation" in d["value"] or "translate" in d["value"]:
            return True
    return False

def make_translation_prompt(data,tokenizer,src:str=None, tgt:str=None):

  lang_dict={"korean":"한국어","english":"영어","ko":"한국어","eng":"영어","en":"영어"}
  src_tgt_dict={"en":"english","eng":"english","english":"english","ko":"korean","kor":"korean","korean":"korean"}

  if not src and not tgt:
    if "src" in data.keys() and "tgt" in data.keys():
      src=src_tgt_dict[data["src"]]
      tgt=src_tgt_dict[data["tgt"]]
    else:
      raise Exception("'src'와 'tgt'가 주어지거나, data의 key로 존재해야합니다.")

  if "term_dict" not in data.keys() or not len(data["term_dict"]):
    template = f"""### Instruction:
{lang_dict[src]}를 {lang_dict[tgt]}로 번역하시오.
### Input:
{data[src]}
### Output:
{data[tgt]}{tokenizer.eos_token}
"""
  else:
    template = f"""### Instruction:
아래의 용어사전을 참조하여, {lang_dict[src]}를 {lang_dict[tgt]}로 번역하시오.

용어사전 : {data["term_dict"]}
### Input:
{data[src]}
### Output:
{data[tgt]}{tokenizer.eos_token}
"""

  return {"text":template}


def add_src_tgt_tag(dataset):

    # split ko->eng and eng->ko
    cut_off=len(dataset)//2
    src_tags=["korean"]*cut_off + ["english"]*(len(dataset)-cut_off)
    tgt_tags=["english"]*cut_off + ["korean"]*(len(dataset)-cut_off)
    dataset=dataset.add_column("src",src_tags)
    dataset=dataset.add_column("tgt",tgt_tags)
    return dataset

def merge_and_resort(df1,df2):
    '''
    df1 : origianl dataset
    df2 : sampled dataset with term_dict
    '''

    def invert_dict(term_dict,src):
        if term_dict is not np.nan:
            if src=="korean" or src== "ko":
                return str(term_dict)
            elif src=="english" or src == "english" :
                return str({v:k for k,v in term_dict.items()})
        else:
            return ""
    
    merged_df=pd.merge(df1,df2,on="id",how="left").fillna({"maps":""})
    merged_df=merged_df.sample(frac=1).reset_index(drop=True)

    term_dict=[invert_dict(t_d,s) for t_d,s in zip(merged_df["term_dict"].values,merged_df["src"].values)]
    merged_df["term_dict"]=term_dict

    return merged_df


def filter_valid_data(data):

    errors=[]
    valid_list=[]
    
    for idx,d in enumerate(data):
        try:
            word_dict=json.loads(d["maps"])
            valid_list.append({"id":d["id"],
                            "term_dict":word_dict})
        except:
            errors.append(idx)
    return valid_list,errors


def prepare_translation_dataset(raw_dataset_path,term_dict_path):
    print("Preparing Dataset")
    dataset=Dataset.load_from_disk(raw_dataset_path)
    dataset=add_src_tgt_tag(dataset).shuffle()
    df1=pd.DataFrame(dataset)

    term_dict_data=[]
    with jsonlines.open(term_dict_path) as f:
        for line in f:
            term_dict_data.append(line)

    valid_data=filter_valid_data(term_dict_data)[0]
    df2=pd.DataFrame(valid_data).sample(frac=1).iloc[:int(len(df1)*0.33)] # dataset의 1/3만 사용 
    merged_df=merge_and_resort(df1,df2)
    merged_df=merged_df.drop_duplicates(subset=["id"])
    dataset=Dataset.from_pandas(merged_df)
    print("Dataset prepared")
    return dataset
    
    
    
    


