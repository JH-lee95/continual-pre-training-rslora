from dataclasses import dataclass

# @dataclass
# class TranslationTemplate:
#     translation_template_w_glossary='''You are a professional translator. Translate the following {} source texts into {}. Refer to the word pairs in the glossary if the glossary exists when you translate. Do not translate the glossary itself.
# {}
# Translation:
# '''

#     translation_template_wo_glossary='''You are a professional translator. Translate the following {} source texts into {}.
# {}
# Translation:
# '''

#     response_template="Translation:"
#     glossary_template="Glossary:"
#     sentence_template="Source:"




@dataclass
class TranslationTemplate:
    translation_template_w_glossary='''You are a professional translator.
Human: Translate the following {} source texts into {}. Refer to the word pairs in the glossary when you translate. Do not translate the glossary itself. Consider the context among the source texts carefully. Do not explain, just translate.
{}
Assistant:
'''

    translation_template_wo_glossary='''You are a professional translator.
Human: Translate the following {} source texts into {}. Consider the context among the source texts carefully. Do not explain, just translate.
{}
Assistant:
'''

    response_template="Assistant:"
    glossary_template="Glossary:"
    sentence_template="Source:"

