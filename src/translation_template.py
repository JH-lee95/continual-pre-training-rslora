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