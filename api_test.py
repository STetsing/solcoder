import requests
import os

API_URL = "https://api-inference.huggingface.co/models/Pipper/SolCoder"
headers = {"Authorization": "Bearer " +  os.environ.get("HF_BEARER_TK")}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "add 2 numbers and return result using math library",
	"aggregation_strategy": "max"
})

print(output)