from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import ABC
import pdb
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    try:
    # in-context example for few-shot generating question
        return f"{entry['text']}"
    except:
        pdb.set_trace()


@field_getter.add("a")
def get_a(entry):
    return entry['summary']


@field_getter.add("qa")
def get_qa(entry):
    # print(entry)
    # pdb.set_trace()
    return f"I will first show a news article in English and then provide a summary of it in the {entry["language"]} language.\n Summarize the following article: {get_q(entry)} \n Summary: {get_a(entry)}"
 

@field_getter.add("gen_a")
def get_gen_a_instruction(entry):
    prompt = "{ice_prompt} \n Learn from the given examples and summarize the article in {language} language.\n Summarize the following article: {question} \n Summary:"
    prompt = prompt.format(
        question=get_q(entry),
        ice_prompt="{ice_prompt}",
        language=entry["language"]
        )
    return prompt




class DatasetWrapper(ABC):
    name = "cross_sum"
    ice_separator = "\n"
    question_field = "text"
    answer_field = "summary"
    hf_dataset = "src/hf_datasets/cross_sum.py"
    hf_dataset_name = 'cross_sum'
    field_getter = field_getter
    
