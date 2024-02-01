from threading import Thread
from typing import Iterator
import gradio as gr
import torch
device = "cuda:2" if torch.cuda.is_available() else 'cpu'

from transformers import BitsAndBytesConfig, StoppingCriteria, AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteriaList

# DEFAULT_SYSTEM_PROMPT = """\
# You are a helpful, respectful and honest assistant with a deep knowledge of Solidity code and software design. 
# Always answer as helpfully as possible, while being safe. Your main programming language is Solidity.
# You only return code if explanations are not explicitely requested.\
# """

DEFAULT_SYSTEM_PROMPT = """You are a AI Coding Assitant with a deep knowledge of Solidity code and software design. Your task is to solve Solidity coding problems and return code snippets based on given user's query. Below is the user's query.
You just return the helpful code."""


model_id = 'codellama/CodeLlama-13b-Instruct-hf'
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)

if torch.cuda.is_available():
    config = AutoConfig.from_pretrained(model_id)
    config.pretraining_tp = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map=device,
        use_safetensors=False,
    )
else:
    model = None

print('INFO: Model size is:', model.num_parameters())
tokenizer = AutoTokenizer.from_pretrained(model_id)

def remove_after_last_occurrence(source, char):
    last_occurrence_index = source.rfind(char)

    if last_occurrence_index != -1:
        result = source[:last_occurrence_index+2]
        return result
    else:
        return source

def get_string_between(source, start_str, end_str):
    start_index = source.find(start_str)
    if start_index == -1:
        return None

    start_index += len(start_str)
    end_index = source.find(end_str, start_index)

    if end_index == -1:
        return None

    return source[start_index:end_index]

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
                if torch.equal(input_ids[0], self.stops[0]):
                    return True 

        for stop in self.stops:
            #print(stop, "in inputs:", is_tensor_at_end(input_ids[0], stop))
            #print('In code value:', self.in_code)
            if is_tensor_at_end(input_ids[0], stop):
                self.in_code -= 1
                
                return True if self.in_code<=0 else False
        return False

def stopping_criteria():
    stop_words = ["\n```"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    return stopping_criteria

def get_prompt(message: str, context: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n""\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    if len(context):
        texts.append(f'Using this context ```\n{context}\n``` ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)


def get_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> int:
    prompt = get_prompt(message)
    input_ids = tokenizer([prompt], return_tensors='np', add_special_tokens=False)['input_ids']
    return input_ids.shape[-1]


def run(context: str,
        comment: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 50) -> Iterator[str]:
    prompt = get_prompt(comment, context)
    stop = stopping_criteria()
    print('INFO: model input:', prompt)
    inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to(device)

    streamer = TextIteratorStreamer(tokenizer,
                                    timeout=10.,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([stop])
    )

    # outputs = model.generate(**generate_kwargs)
    # text = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    
    # parsed_text = get_string_between(text, "```", "```")
    # result = parsed_text if parsed_text is not None else text
    # return remove_after_last_occurrence(result, '}')

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield ''.join(outputs)

app = gr.Interface(
    fn=run,
    inputs=["text", "text", gr.Slider(0, 2000,300), gr.Slider(0.01, 1, 0.2),
            gr.Slider(0, 1, 0.9), gr.Slider(1, 200, 50)],
    outputs=["text"],
    title="Llama 13B",
    allow_flagging="manual",
    flagging_options=["wrong answer", "off topic"]
)


if __name__ == "__main__":
    app.launch(share=True)
