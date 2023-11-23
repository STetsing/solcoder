import requests
import os

API_URL = "https://api-inference.huggingface.co/models/Pipper/sol_processed_s2s"
headers = {"Authorization": "Bearer " +  os.environ.get("HF_BEARER_TK")}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "get the vote winner",
})

print(output)