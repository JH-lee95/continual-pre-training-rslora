from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
import torch
import ast
import ipdb
import json
import requests
from peft import LoraConfig, PeftModel
from typing import Optional
from dataclasses import dataclass
from typing import Union
import ipdb


@dataclass
class TranslationTemplate:
    translation_template_w_glossary='''Translate the following {} source text into {}. Refer to the word pairs in the glossary when you translate. Do not translate the glossary itself. 
{}
'''

    translation_template_wo_glossary='''Translate the following {} source text into {}.
{}
'''

    response_template="### Target:"
    glossary_template="### Glossary:"
    sentence_template="### Source:"
    system_prompt="You are a professional translator."


def gen(model,prompt):
    generation_config = GenerationConfig(
        temperature=0.15,
        # top_p=0.9,
        max_new_tokens=512,
        repetition_penalty=1.02,
        # num_beams=3,
        # num_return_sequences=3,
        stop_token_ids=[tokenizer.eos_token_id].extend(tokenizer.encode(tokenizer.tokenize("<|eot_id|>")[0],add_special_tokens=False)),
        do_sample=True,
    )
    gened = model.generate(
        **tokenizer(
            prompt,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        generation_config=generation_config,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


    gen_result=[]
    for gen in gened:
        result_str = tokenizer.decode(gen)
        gen_result.append(result_str)
    return gen_result

def formatting_glossary(term_dict,glossary_template):
    glossary=[f"{k}={v}" for k,v in term_dict.items()]
    glossary_str="\n".join(glossary)
    glossary_str=f"{glossary_template}\n{glossary_str}".strip()

    return glossary_str




base_path="/nvme0/models--mirlab--AkaLlama-llama3-70b-v0.1/snapshots/c94416f4753580a13a5f195217d0f29e1d9a606f"
peft_path="/azurestorage/models/trained/llama3-70b-inst_2024-05-09-16-51-1715241061/checkpoint-120/"


model=AutoModelForCausalLM.from_pretrained(base_path,device_map="auto")
tokenizer=AutoTokenizer.from_pretrained(peft_path)

peft_model=PeftModel.from_pretrained(model,peft_path)


text='''서면국제세원- 2908, 2016.05.17
 
국내사업장이 없는 미국법인이 내국법인에게 기본설계용역을 제공하고 수취하는 소득은 동 용역제공대가가 비공개 기술, 정보 등 노하우의 사용대가인 경우에 사용료소득에 해당하는 것이나, 동 용역제공대가가 동종의 용역수행자가 통상적으로 보유하는 전문지식이나 기능을 활용하여 수행하는 용역에 대한 대가인 경우에는 사업소득에 해당되는 것임. 동 용역제공대가가 사용료소득인지 사업소득인지의 여부는 내국법인이 제공받는 용역의 실질내용에 따라 사실 판단할 사항이며, 법인세법 기본통칙 93-132…7 [노하우와 독립적인 인적용역의 구분]을 참고하기 바람.
'''.strip()


term_dict=""

src="Korean"
tgt="English"

if len(term_dict):
    term_dict = ast.literal_eval(term_dict)
    term_dict=formatting_glossary(term_dict,TranslationTemplate.glossary_template)
    text=f"{term_dict}\n{TranslationTemplate.sentence_template}\n{text}"
    prompt=TranslationTemplate.translation_template_w_glossary.format(src,tgt,text)

else:
    text=f"{TranslationTemplate.sentence_template}\n{text}"
    prompt=TranslationTemplate.translation_template_wo_glossary.format(src,tgt,text)
    

print(prompt)



messages = [
    {"role": "system", "content": TranslationTemplate.system_prompt},
    {"role": "user", "content": prompt},
]

template=tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                )
template+="<|start_header_id|>assistant<|end_header_id|>\n###Target:\n"


result=gen(model,template)
print(result[0])

ipdb.set_trace()