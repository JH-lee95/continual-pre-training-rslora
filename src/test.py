from comet import download_model, load_from_checkpoint
from datasets import load_dataset,Dataset
import datasets
from vllm import LLM, SamplingParams
from utils import make_translation_prompt
from transformers import AutoTokenizer
import argparse
import sys,os
import ipdb
import json


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_dir",type=str,required=True,help="local or huggingface hub directory of base model")
    parser.add_argument("--comet_model_dir",type=str,default="Unbabel/wmt22-comet-da")
    parser.add_argument("--log_dir",type=str, default=None)

    return parser.parse_args()


def load_models(model_path,comet_model_path):

    comet_model = load_from_checkpoint(download_model(comet_model_path))
    model = LLM(model=model_path)  # Create an LLM.
    # model=None
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    return model,tokenizer, comet_model


def eval_translation_scores(model,tokenizer,comet_model,dataset,sampling_params=None):
    """
    model : vllm llm instance
    """
    sacrebleu = datasets.load_metric('sacrebleu')

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

    pred_ko_to_eng=model.generate(input_ko_to_eng,sampling_params)
    pred_eng_to_ko=model.generate(input_eng_to_ko,sampling_params)

    pred_ko_to_eng=[request.outputs[0].text.strip() for request in pred_ko_to_eng]
    pred_eng_to_ko=[request.outputs[0].text.strip() for request in pred_eng_to_ko]

    comet_ko_to_eng=[]
    comet_eng_to_ko=[]
    for s_ko,r_eng,p_ko_to_eng,s_eng,r_ko,p_eng_to_ko in zip(sources_ko,references_eng,pred_ko_to_eng,sources_eng,references_ko,pred_eng_to_ko):
        comet_ko_to_eng.append({
            "src": s_ko,
            "mt": p_ko_to_eng,
            "ref": r_eng[0]
        })

        comet_eng_to_ko.append({
            "src": s_eng,
            "mt": p_eng_to_ko,
            "ref": r_ko[0]
        })

    sacre_bleu_ko_to_eng= sacrebleu.compute(predictions=pred_ko_to_eng, references=references_eng)['score']
    sacre_bleu_eng_to_ko= sacrebleu.compute(predictions=pred_eng_to_ko, references=references_ko)['score']

    comet_score_ko_to_eng = comet_model.predict(comet_ko_to_eng, batch_size=32, gpus=1)[1]
    comet_score_eng_to_ko = comet_model.predict(comet_eng_to_ko, batch_size=32, gpus=1)[1]

    return {"sacre_bleu_ko_to_eng":sacre_bleu_ko_to_eng,
            "sacre_bleu_eng_to_ko":sacre_bleu_eng_to_ko,
            "comet_score_ko_to_eng":comet_score_ko_to_eng,
            "comet_score_eng_to_ko":comet_score_eng_to_ko,}


def main(args):

    sampling_params = SamplingParams(temperature=0.35,max_tokens=4096)
    model,tokenizer,comet_model=load_models(args.base_model_dir,args.comet_model_dir)

    dataset_wo_term_dict=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/prepared_for_training/translation_dataset_valid_wo_term_dict").select(range(1000))
    dataset_w_term_dict=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/prepared_for_training/translation_dataset_valid_w_term_dict").select(range(1000))
    flores=Dataset.load_from_disk("/root/azurestorage/data/번역데이터셋/aligned_dataset/prepared_for_training/flores_ko_eng/test")

    score_wo_term_dict=eval_translation_scores(model,tokenizer,comet_model,dataset_wo_term_dict,sampling_params=sampling_params)
    score_w_term_dict=eval_translation_scores(model,tokenizer,comet_model,dataset_w_term_dict,sampling_params=sampling_params)
    score_flores=eval_translation_scores(model,tokenizer,comet_model,flores,sampling_params=sampling_params)

    print("----score_wo_term_dict----")
    print(score_wo_term_dict)
    print("----score_w_term_dict----")
    print(score_w_term_dict)
    print("----score_flores----")
    print(score_flores)

    if not os.path.exists("../test_result/"):
        os.mkdir("../test_result/")

    with open(f'''../test_result/{args.base_model_dir.split("/")[-1]}.jsonl''', "w",encoding='utf-8') as f:
        f.write(json.dumps({"score_wo_term_dict":score_wo_term_dict},ensure_ascii=False)+"\n")
        f.write(json.dumps({"score_w_term_dict":score_w_term_dict},ensure_ascii=False)+"\n")
        f.write(json.dumps({"score_flores":score_flores},ensure_ascii=False)+"\n")


if __name__=="__main__":
    args=parse_args()
    main(args)