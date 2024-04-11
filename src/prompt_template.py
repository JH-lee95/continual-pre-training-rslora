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

    post_editor_system_definition="""{} source : {}

{} translation : {}

You are a professional post editor and translator of an accounting firm. According to the client's request, the result of translating the original {} text into {} is the same as above. You carry out the request of the client related to editing and translation on the selected text below. Just answer the question without explanation.    

selected text : {}
"""# source lang, source text, target lang,target text,source lang, target lang,selected text

    sentence_template="### Sentence:"
    glossary_template="### Glossary:"
    glossary_tags=["."]
    translation_sent2term='''### Instruction:
Translate the following {} sentences into {}, referring to the word pairs in the glossary if the glossary exists after each sentence finishes. Do not translate the glossary it self.
{}
### Translation:
'''
