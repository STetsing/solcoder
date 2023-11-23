import json, requests


headers = {'Content-Type': 'application/json'}
scoring_uri ="http://127.0.0.1:8080/infer"


test_data = json.dumps({'comment': "Calls winningProposal() to determine the voter winner. @return winnerName_ the name of the winner"})
print('sending the request', '...')
#response = infer(test_data)
response = requests.post(scoring_uri, data=test_data, headers=headers)
data = json.loads(response.json())
print(data)