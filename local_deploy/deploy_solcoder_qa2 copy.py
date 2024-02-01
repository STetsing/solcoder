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
                                            #load_in_4bit=True,
                                            trust_remote_code=True,
                                            device_map=device)

print(model.get_memory_footprint()/1e9)
model =  PeftModel.from_pretrained(model, model_path)
model = model.merge_and_unload()
print(model.get_memory_footprint()/1e9)

def is_tensor_at_end(main_tensor, sub_tensor):
    # Convert scalar tensor to tensor
    if torch.is_tensor(sub_tensor):
        sub_tensor = sub_tensor.view(-1)
    else:
        sub_tensor = torch.tensor([sub_tensor])
    # Check if sub_tensor is at the end of main_tensor
    if sub_tensor.size() > main_tensor.size():
        return False
    else:
        return torch.all(main_tensor[-len(sub_tensor):] == sub_tensor)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to(device) for stop in stops]
        self.in_code = 0
        incode_words = ["{\n", " {\n", "\n{\n"]
        self.in_code_token = [tokenizer(icw, return_tensors='pt')['input_ids'].squeeze().to(device) for icw in incode_words]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        #print('#'*100)
        #print('new inputs: ', input_ids)
        for icw in self.in_code_token:
            if is_tensor_at_end(input_ids[0], icw):
                self.in_code += 1

        for stop in self.stops:
            #print(stop, "in inputs:", is_tensor_at_end(input_ids[0], stop))
            #print('In code value:', self.in_code)
            if is_tensor_at_end(input_ids[0], stop):
                self.in_code -= 1
                return True if self.in_code==0 else False
        return False

def stopping_criteria():
    stop_words = ["\nQuestion:", "}\n", "User:", "INSTRUCTION:", "INPUT:", "A:", "Instruction:", "Output:"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
    print('Stop ids:', stop_words_ids)
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    return stopping_criteria

def infer(comment, mnt=200, temp=0.8):
    stop = stopping_criteria()
    comment = "### Question: // Write in solidity "+ comment + "\nAnswer:\n"
    model_inputs = tokenizer([comment], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=mnt,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=temp,
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

app = gr.Interface(
    fn=infer,
    inputs=["text", gr.Slider(0, 500,100), gr.Slider(0, 1, 0.8)],
    outputs=["text"],
    allow_flagging="manual",
    flagging_options=["wrong answer", "off topic"]
)


if __name__ == "__main__":
    app.launch(share=True)
