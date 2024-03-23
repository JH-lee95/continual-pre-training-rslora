import os,sys
import random
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset,Dataset,concatenate_datasets
import jsonlines
import ipdb
import json
from kiwipiepy import Kiwi
from nltk.tokenize import sent_tokenize
import ast
import ipdb

kiwi = Kiwi()

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def make_translation_input_from_dataset(data,
                                  tokenizer,
                                  translation_template_wo_term_dict:str=None,
                                  translation_template_w_term_dict:str=None,
                                  glossary_template:str=None,
                                  glossary_tags:list=None,
                                  src:str=None, 
                                  tgt:str=None,
                                  output=True):

  lang_dict={"korean":"Korean","english":"English","ko":"Korean","eng":"English","en":"English"}
  src_tgt_dict={"en":"english","eng":"english","english":"english","ko":"korean","kor":"korean","korean":"korean"}

  if not src and not tgt:
    if "src" in data.keys() and "tgt" in data.keys():
      src=src_tgt_dict[data["src"]]
      tgt=src_tgt_dict[data["tgt"]]
    else:
      raise Exception("'src'와 'tgt'가 주어지거나, data의 key로 존재해야합니다.")

  if glossary_template is not None and glossary_tags is not None:
    text=pair_sent_terms(lang=src,
                        text=data[src],
                        term_dict=data["term_dict"],
                        glossary_template=glossary_template,
                        glossary_tags=glossary_tags,)
    template=translation_template_wo_term_dict.format(lang_dict[src],lang_dict[tgt],text)

  else:
    text=data[src]
    term_dict=data["term_dict"]
    if len(term_dict) and term_dict is not None:
        if translation_template_w_term_dict is not None:
            template=translation_template_w_term_dict.format(lang_dict[src],lang_dict[tgt],term_dict,text)
        else:
            raise "translation_template_w_term_dict is not given."
    else:
        template=translation_template_wo_term_dict.format(lang_dict[src],lang_dict[tgt],text)


  if output:
      template=template+data[tgt]+tokenizer.eos_token
      
  return {"text":template}

def pair_sent_terms(lang,
                    text,
                    term_dict:str,
                    glossary_template: str,
                    glossary_tags: list[str,str],):

    lang_dict={"korean":"korean","ko":"korean","kor":"korean","eng":"english","english":"english","en":"english"}
    src = lang_dict[lang]
    
    splited_sents=[]
    paras=text.split("\n") #split text into paragraphs based on linebreak to keep its original format.
    
    for idx,para in enumerate(paras):
        if len(para.strip()):
            if src=="korean":
                temp_sents=[s.text for s in kiwi.split_into_sents(para)]
            else:
                temp_sents=sent_tokenize(para)
            if idx<len(paras)-1:
                temp_sents[-1]+="\n" #keep linebreak
            splited_sents.extend(temp_sents)
        else:
            splited_sents[-1]+="\n"
            

    sent2terms = []
    if len(term_dict):
        term_dict = ast.literal_eval(term_dict)
        for s in splited_sents:
            new_sent_parts = {}
            for k, v in term_dict.items():
                if k in s:
                    new_sent_parts[k]=v

            if len(new_sent_parts):
                new_s = "### Sentence:"+s + "\n" + "### Glossary:" + str(new_sent_parts) +"\n"
            else:
                new_s = "### Sentence:"+s + "\n" + "### Glossary:" + "\n"
            sent2terms.append(new_s)
    else:
        # Handle case of empty term_dict (e.g., directly append sentences)
        for s in splited_sents:
            new_s = "### Sentence:"+s + "\n" + "### Glossary:" + "\n"
            sent2terms.append(new_s)

    return "".join(sent2terms).rstrip()


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
