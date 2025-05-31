from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import ABC
import pdb
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    try:
    # in-context example for few-shot generating question
        return f"{entry['context']} \t Question: {entry['question']}"
    except:
        pdb.set_trace()


@field_getter.add("a")
def get_a(entry):
    return entry['answer']


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)} \n Answer: {get_a(entry)}"
   

@field_getter.add("gen_a")
def get_gen_a_instruction(entry):
    prompt = "{ice_prompt} \n Learn from the given examples and answer the next question in {language} language. \n {question} \n Answer:"
    prompt = prompt.format(
        question=get_q(entry),
        ice_prompt="{ice_prompt}",
        language=entry["language"]
        )
    return prompt


class DatasetWrapper(ABC):
    name = "xquad"
    ice_separator = "\n"
    question_field = "question"
    answer_field = "answer"
    hf_dataset = "src/hf_datasets/xquad.py"
    hf_dataset_name = 'xquad'
    field_getter = field_getter
    
