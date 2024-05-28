from dataclasses import dataclass

@dataclass
class TranslationTemplate:
    translation_template_w_glossary='''Translate the following {} source text into {}. Refer to the word pairs in the glossary when you translate. Do not translate the glossary itself.
{}
'''

    translation_template_wo_glossary='''Translate the following {} source text into {}.
{}
'''
    system_prompt="You are a professional translator. You are especially familiar with specialized knowledge and terms in economics, law, and accounting, as well as general everyday terms."
    # system_prompt=None
    # response_template="### Target:"
    response_template="<|CHATBOT_TOKEN|>"
    glossary_template="### Glossary:"
    sentence_template="### Source:"
    
    ## default chat_template
    # chat_template="{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message + '\n'}}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ content }}{% elif message['role'] == 'assistant' %}{{ content + '\\n' }}{% endif %}{% endfor %}"

    ## llama3 
    # chat_template="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    ## eeve inst
    # chat_template="{% for message in messages %}{% if message['role'] == 'user' %}{{'Human: ' + message['content'].strip() + '\n'}}{% elif message['role'] == 'system' %}{{message['content'].strip()+ '\n'}}{% elif message['role'] == 'assistant' %}{{ 'Assistant:\n'  + message['content']}}{% endif %}{% endfor %}"

    ## chatml
    # chat_template="{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}"

    chat_template=None

# @dataclass
# class TranslationTemplate:
#     translation_template_w_glossary='''You are a professional translator. Translate the following {} texts into {}, referring to the word pairs in the glossary if the glossary exists after each text finishes. Do not translate the glossary itself.
# {}
# Translation:
# '''

#     translation_template_wo_glossary=translation_template_w_glossary

#     response_template="Translation:"
#     glossary_template="Glossary:"
#     sentence_template="Sentence:"




# @dataclass
# class TranslationTemplate:
#     translation_template_w_glossary='''You are a professional translator.
# Human: Translate the following {} source texts into {}. Refer to the word pairs in the glossary when you translate. Do not translate the glossary itself. Do not explain, just translate.
# {}
# Assistant:
# '''

#     translation_template_wo_glossary='''You are a professional translator.
# Human: Translate the following {} source texts into {}. Do not explain, just translate.
# {}
# Assistant:
# '''

#     response_template="Assistant:"
#     glossary_template="Glossary:"
#     sentence_template="Source:"

