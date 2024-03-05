from comet import download_model, load_from_checkpoint
from datasets import load_dataset,Dataset
import datasets
from transformers import AutoTokenizer
import argparse
import sys,os
import ipdb
import json
import pandas as pd
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
import sglang as sgl



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--comet_model_dir",type=str,default="Unbabel/wmt22-comet-da")
    parser.add_argument("--model_path",type=str,default=None)
    parser.add_argument("--log_dir",type=str, default=None)

    return parser.parse_args()

@sgl.function
def translation_inference(s, prompt:str,**kwargs):

    s += prompt + sgl.gen("output",**kwargs)



def make_translation_prompt(src:str,tgt:str,text:str,term_dict:dict):

    if term_dict is not None:
        template=f"""<|im_start|>system 
Translate the {src} sentence into {tgt}.
Glossary :  {term_dict}
<|im_end|> 
<|im_start|>user 
{text}
<|im_end|> 
<|im_start|>assistant
"""

    else:
        template= f"""<|im_start|>system 
Translate the {src} sentence into {tgt}, referring to the glossary below.
<|im_end|> 
<|im_start|>user 
{text}
<|im_end|> 
<|im_start|>assistant
"""
    return template


def prepare_test_dataset(dataset):
    src_tgt_dict={"en":"english","eng":"english","english":"english","ko":"korean","kor":"korean","korean":"korean"}

    sources_ko=[]
    references_eng=[]
    sources_eng=[]
    references_ko=[]
    input_ko_to_eng=[]
    input_eng_to_ko=[]

    for data in dataset:

        if src_tgt_dict[data["src"]]=="korean":
            sources_ko.append(data["korean"])
            references_eng.append([data["english"]])
            input_ko_to_eng.append(make_translation_prompt(data,tokenizer,no_output=True)["text"])
        else:
            sources_eng.append(data["english"])
            references_ko.append([data["korean"]])
            input_eng_to_ko.append(make_translation_prompt(data,tokenizer,no_output=True)["text"])

    return  sources_ko,references_eng,sources_eng,references_ko,input_ko_to_eng,input_eng_to_ko


def get_metrics(src,ref,hyp,comet_model):
    sacrebleu = datasets.load_metric('sacrebleu')
    bleu_score=sacrebleu.compute(predictions=hyp, references=[[r] for r in ref])['score']

    comet_input=[{"src": s,
                "mt": r,
                "ref": h,
                } for s,r,h in zip(src,ref,hyp)]
    comet_score=comet_model.predict(comet_input, batch_size=32, gpus=1)[1]
    
    return bleu_score,comet_score

# def gen_and_eval(tokenizer,comet_model,dataset):

#     sources_ko,references_eng,sources_eng,references_ko,input_ko_to_eng,input_eng_to_ko=prepare_test_dataset(dataset)

#     pred_ko_to_eng=translation_inference(input_ko_to_eng,temperature=0.15,max_new_tokens=4096)
#     pred_eng_to_ko=translation_inference(input_eng_to_ko,temperature=0.15,max_new_tokens=4096)

#     pred_ko_to_eng=[output["output"] for output in pred_ko_to_eng]
#     pred_eng_to_ko=[output["output"] for output in pred_eng_to_ko]

#     sacre_bleu_ko_to_eng,comet_score_ko_to_eng=get_metrics(soureces_eng,references_ko,pred_eng_to_ko)
#     sacre_bleu_eng_to_ko,comet_score_eng_to_ko=get_metrics(soureces_eng,references_ko,pred_eng_to_ko)

#     return {"sacre_bleu_ko_to_eng":sacre_bleu_ko_to_eng,
#             "sacre_bleu_eng_to_ko":sacre_bleu_eng_to_ko,
#             "comet_score_ko_to_eng":comet_score_ko_to_eng,
#             "comet_score_eng_to_ko":comet_score_eng_to_ko,}


def main(args):

    comet_model = load_from_checkpoint(download_model(args.comet_model_dir))
    tokenizer=AutoTokenizer.from_pretrained(args.model_path)


    df=pd.read_excel("/root/azurestorage/data/번역데이터셋/raw_data/test_data_blood,sweat,tear.xlsx")

    kor,eng=df["Korean"].values.tolist(),df["English"].values.tolist() 
    kor_to_eng=[{"prompt":make_translation_prompt(src="한국어",tgt="영어",text=text,term_dict=None)} for text in kor]
    eng_to_ko=[{"prompt":make_translation_prompt(src="영어",tgt="한국어",text=text,term_dict=None)} for text in eng]

    kor_to_eng_gen=translation_inference.run_batch(kor_to_eng,progress_bar=True,temperature=0.15,max_new_tokens=4096)
    kor_to_eng_gen=[t.text().split("<|im_start|>assistant\n")[-1] for t in kor_to_eng_gen]

    eng_to_ko_gen=translation_inference.run_batch(eng_to_ko,progress_bar=True,temperature=0.15,max_new_tokens=4096)
    eng_to_ko_gen=[t.text().split("<|im_start|>assistant\n")[-1] for t in eng_to_ko_gen]

    our_model_k_bleu,our_model_k_comet=get_metrics(eng,kor,eng_to_ko_gen,comet_model)
    our_model_e_bleu,our_model_e_comet=get_metrics(kor,eng,kor_to_eng_gen,comet_model)
    
    print("----our_model_e_bleu----")
    print(our_model_e_bleu)
    print("----our_model_e_comet----")
    print(our_model_e_comet)
    print("----our_model_k_bleu----")
    print(our_model_k_bleu)
    print("----our_model_k_comet----")
    print(our_model_k_comet)

    # if not os.path.exists("../test_result/"):
    #     os.mkdir("../test_result/")

    # with open(f'''../test_result/{args.base_model_dir.split("/")[-1]}.jsonl''', "w",encoding='utf-8') as f:
    #     f.write(json.dumps({"score_wo_term_dict":score_wo_term_dict},ensure_ascii=False)+"\n")
    #     f.write(json.dumps({"score_w_term_dict":score_w_term_dict},ensure_ascii=False)+"\n")
    #     f.write(json.dumps({"score_flores":score_flores},ensure_ascii=False)+"\n")




# def main(args):

#     comet_model = load_from_checkpoint(download_model(args.comet_model_path))
#     tokenizer=AutoTokenizer.from_pretrained(args.model_path)

#     dataset_wo_term_dict=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/prepared_for_training/translation_dataset_valid_wo_term_dict").select(range(1000))
#     dataset_w_term_dict=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/prepared_for_training/translation_dataset_valid_w_term_dict").select(range(1000))
#     flores=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/prepared_for_training/flores_ko_eng/test")

#     score_wo_term_dict=gen_and_eval(tokenizer,comet_model,dataset_wo_term_dict)
#     score_w_term_dict=gen_and_eval(tokenizer,comet_model,dataset_w_term_dict)
#     score_flores=gen_and_eval(tokenizer,comet_model,flores)

#     print("----score_wo_term_dict----")
#     print(score_wo_term_dict)
#     print("----score_w_term_dict----")
#     print(score_w_term_dict)
#     print("----score_flores----")
#     print(score_flores)

#     if not os.path.exists("../test_result/"):
#         os.mkdir("../test_result/")

#     with open(f'''../test_result/{args.base_model_dir.split("/")[-1]}.jsonl''', "w",encoding='utf-8') as f:
#         f.write(json.dumps({"score_wo_term_dict":score_wo_term_dict},ensure_ascii=False)+"\n")
#         f.write(json.dumps({"score_w_term_dict":score_w_term_dict},ensure_ascii=False)+"\n")
#         f.write(json.dumps({"score_flores":score_flores},ensure_ascii=False)+"\n")


if __name__=="__main__":
    set_default_backend(RuntimeEndpoint("http://localhost:30000"))
    args=parse_args()
    main(args)