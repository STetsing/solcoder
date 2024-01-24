# SolCoder
SolCoder is LLM which generates solidity source code from user prompt comments. It is hosted on the Huggingface repository [Pipper/Solcoder](https://huggingface.co/Pipper/SolCoder) and is continuously uptated. The model output is focussed on generating functions from user comments.

P.S.: No source code context is considered if using an editor or such. The generated code might not be executable due to context variables and other dependencies, that might be defined prior to the generation. The user might adjust these to be aligned with the context.


## Dataset Sources for Training & Finetuning 
The datasets used is an aggregation of multiple datasources.

### Sourcify
You can follow the [instructions in the docs](https://docs.sourcify.dev/docs/repository/#s3-bucket) and contact [Kaan Uzdogan](mailto:kaan.uzdogan@ethereum.org) for the credentials.

See [slither-solidity](https://github.com/STetsing/slither-solidity.git) for the sourcify data processing.

### Slither audited smart contracts
[mwritescode/slither-audited-smart-contracts](https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts)

### StarCoder 
[bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup)

```
from datasets import load_dataset

# specific language (e.g. Dockerfiles)
ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/solidity", split="train")
```

## Audit and Security
IMPORTANT: None of the code parsed in the dataset is statically audited or further audited if already. Use static analyzers such as [slither](https://github.com/crytic/slither#api-documentation) for the purpose. 

## Setup the Hugginface Tokens 
Important: The token must have write access. This is use to push data and model to the HF hub automatically at creation time
``` 
export HF_TOKEN="your write access token"
export HF_BEARER_TK="your api bearer token from HF"
```

## Launch accelerated training 
> accelerate launch --num_cpu_threads_per_process 8  train/train_phi2_lora_qa.py 