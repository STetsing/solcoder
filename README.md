# Solcoder




## Dataset sources

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
IMPORTANT: None of the code parsed in the dataset is statically audited or further audited if already. Use static analyzers such as [slither](https://github.com/crytic/slither#api-documentation)

## Setup the hugginface token 
Important: The token must have write access
``` 
export HF_TOKEN="your write access token"
```