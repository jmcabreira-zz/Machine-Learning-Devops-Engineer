import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# load config file
with open('config.json') as f:
  config = json.load(f)

output_path = os.path.join(config['output_model_path'], 'apireturns2.txt')

#Call each API endpoint and store the responses
response1 = requests.post(f'{URL}/prediction?datapath=testdata/testdata.csv').content
response2 = requests.get(f'{URL}/scoring?datapath=testdata/testdata.csv').content
response3 = requests.get(f'{URL}/summarystats?datapath=testdata/testdata.csv').content
response4 = requests.get(f'{URL}/diagnostics?datapath=testdata/testdata.csv').content

# print("response 1 :", response1)
# print("response 2 :", response2)
# print("response 3 :", response3)
# print("response 4 :", response4)

#combine all API responses
responses = {
    'Prediction response': json.loads(response1.decode('utf8').replace("'", '"')),
    'Scoring response': json.loads(response2.decode('utf8').replace("'", '"')),
    'Summary stats response': json.loads(response3.decode('utf8').replace("'", '"')),
    'Diagnostics response': json.loads(response4.decode('utf8').replace("'", '"'))
}
# print(responses)

# write the responses to your workspace
with open(output_path, 'w') as f:
    f.write(json.dumps(responses))


