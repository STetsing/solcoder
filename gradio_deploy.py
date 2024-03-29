import gradio as gr
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, RobertaTokenizer, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else 'cpu'

model_path = './SolExplain/checkpoint-287760/'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

def infer(comment, max_new_tokens=200, temperature=0.9, sample=False):
    input_ids = tokenizer(comment, return_tensors='pt').input_ids.to(device)
    outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=sample)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


app = gr.Interface(
    fn=infer,
    inputs=["text", gr.Slider(0, 500,100), gr.Slider(0, 1, 0.8), "checkbox"],
    outputs=["text"],
    allow_flagging="manual",
    flagging_options=["wrong answer", "off topic"]
)


if __name__ == "__main__":
    app.launch(share=True)

# demo = gr.ChatInterface(random_response)

# demo.launch()