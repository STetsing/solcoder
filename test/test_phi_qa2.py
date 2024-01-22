import torch 
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = "cuda:1" if torch.cuda.is_available() else 'cpu'

#model_path = './trained_model_last_epoch'
#model_path = '/Users/pippertetsing/Desktop/work/Remix/solcoder/training_models/checkpoint-720'
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

#print(infer("### Question: Write in Solidity a function _super_adder for adding 6 uint256 numbers and return the result. The function should be internal\n### Answer:\n"))
print(infer("### Question: Write in Solidity a function get_winningProposal to Computes the winning proposal taking all previous votes into account.\n### Answer:\n"))