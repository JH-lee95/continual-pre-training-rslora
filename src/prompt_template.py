from dataclasses import dataclass

@dataclass
class TranslationTemplate:
    translation_template='''You are a professional translator. Translate the following {} source text into {}. Refer to the word pairs in the glossary if the glossary exists when you translate. Do not translate the glossary it self.
{}
Translation:
'''
    response_template="Translation:"
    glossary_template="Glossary:"
    sentence_template="Source:"

