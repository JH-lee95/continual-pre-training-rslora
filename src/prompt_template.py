from dataclasses import dataclass

@dataclass
class TranslationTemplate:
    translation_template_w_glossary='''You are a professional translator. Translate the following {} texts into {}. Refer to the word pairs in the glossary when you translate. You can ignore the words in the glossary that hurt the context and naturality. Do not translate the glossary itself. 
{}
Translation:
'''

    translation_template_wo_glossary='''You are a professional translator. Translate the following {} texts into {}.
{}
Translation:
'''

    response_template="Translation:"
    glossary_template="Glossary:"
    sentence_template="Source:"



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

