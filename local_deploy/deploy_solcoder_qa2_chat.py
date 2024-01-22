import torch 
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from peft import PeftModel
import gradio as gr
from threading import Thread


device = "cuda:1" if torch.cuda.is_available() else 'cpu'

model_path = './Phi2-SolCoder-lora-qa2'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                            load_in_4bit=True,
                                            trust_remote_code=True,
                                            device_map=device)

print(model.get_memory_footprint()/1e9)
model =  PeftModel.from_pretrained(model, model_path)
model = model.merge_and_unload()
print(model.get_memory_footprint()/1e9)

def infer(comment):
    input_ids = tokenizer(comment, return_tensors='pt')#.input_ids.to(device)
    outputs = model.generate(**input_ids, max_new_tokens=200, do_sample=True, temperature=0.8)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# class StopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         stop_ids = [29, 0]
#         for stop_id in stop_ids:
#             if input_ids[0][-1] == stop_id:
#                 return True
#         return False

def is_tensor_at_end(main_tensor, sub_tensor):
    # Convert scalar tensor to tensor
    if torch.is_tensor(sub_tensor):
        sub_tensor = sub_tensor.view(-1)
    else:
        sub_tensor = torch.tensor([sub_tensor])
    # Check if sub_tensor is at the end of main_tensor
    return torch.all(main_tensor[-len(sub_tensor):] == sub_tensor)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to(device) for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        print('#'*100)
        print('new inputs: ', input_ids)
        for stop in self.stops:
            print(stop, "in inputs:", is_tensor_at_end(input_ids[0], stop))
            if is_tensor_at_end(input_ids[0], stop):
                return True
        return False

def stopping_criteria():
    stop_words = ["\nQuestion: ", "}\n", "User:", "INSTRUCTION", "INPUT:", "A:"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
    print('Stop ids:', stop_words_ids)
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    return stopping_criteria

def predict(message, history):

    history_transformer_format = history + [[message, ""]]
    stop = stopping_criteria()

    messages = "".join(["".join(["### Question: // Write in solidity "+item[0], "\nAnswer:"+item[1]])  #curr_system_message +
                for item in history_transformer_format])

    # model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=0.8,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message  = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            yield partial_message


gr.ChatInterface(predict).launch()

#print(infer("### Question: Write in Solidity a function _super_adder for adding 6 uint256 numbers and return the result. The function should be internal\n### Answer:\n"))
#print(infer("### Question: Write in Solidity a function get_winningProposal to Computes the winning proposal taking all previous votes into account.\n### Answer:\n"))