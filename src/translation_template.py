from dataclasses import dataclass

@dataclass
class TranslationTemplate:
    template_w_term_dict='''### Instruction:
Translate the {} text into {}, referring to the glossary below.

Glossary : {}
### Input:
{}
### Translation:
'''

    template_wo_term_dict='''### Instruction:
Translate the {} text into {}.
### Input:
{}
### Translation:
'''

    response_template="### Translation:"


#     template_w_term_dict='''### Instruction:
# 아래의 용어사전을 참조하여, {}를 {}로 번역하시오.

# 용어사전 : {}
# ### Input:
# {}
# ### Output:
# '''

#     template_wo_term_dict='''### Instruction:
# {}를 {}로 번역하시오.
# ### Input:
# {}
# ### Output:
# '''

#     response_template="### Output:"
    
    # source lang, source text, target lang,target text,source lang, target lang,selected text
    post_editor_system_definition="""{} source : {}


{} translation : {}

You are a professional post editor and translator of an accounting firm. According to the client's request, the result of translating the original {} text into {} is the same as above. You carry out the request of the client related to editing and translation on the selected text below. Just answer the question without explanation.    

selected text : {}
"""