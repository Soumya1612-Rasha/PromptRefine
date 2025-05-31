from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import ABC
import pdb
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    try:
    # in-context example for few-shot generating question
        return f"{entry['source']}"
    except:
        pdb.set_trace()


@field_getter.add("a")
def get_a(entry):
    return entry['target']


# @field_getter.add("qa")
# def get_qa(entry):
#     return f"{get_q(entry)}\t{get_a(entry)}"

@field_getter.add("qa")
def get_qa(entry):
    return f"Input: {get_q(entry)} \t Output: {get_a(entry)}"
    # return f"Translate to English: {get_q(entry)}. Translated: {get_a(entry)}"



@field_getter.add("gen_a")
def get_gen_a_instruction(entry):
   
    prompt = "{ice_prompt} \n Translate the following sentence to English:\n Input: {question} \n Output:"
    prompt = prompt.format(
        question=get_q(entry),
        ice_prompt="{ice_prompt}")
    return prompt


class DatasetWrapper(ABC):
    name = "flores"
    ice_separator = "\n"
    question_field = "source"
    answer_field = "target"
    hf_dataset = "src/hf_datasets/flores.py"
    hf_dataset_name = 'flores'
    field_getter = field_getter
    
