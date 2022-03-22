import os
import requests
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

#load config
with open('config.json', 'r') as f1:
    config = json.load(f1)

def main():
    model_path = os.path.join(config['output_model_path'])
    api_return_file_name = config['api_result_name']
    test_data = os.path.join(config['test_data_path'], config['test_data_name'])

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    #Call each API endpoint and store the responses
    response1 = requests.post("{}/prediction".format(URL), json={"dataset_path": test_data}).text
    response2 = requests.get("{}/scoring".format(URL)).text
    response3 = requests.get("{}/summarystats".format(URL)).text
    response4 = requests.get("{}/diagnostics".format(URL)).text

    #combine all API responses
    responses = "{}\n{}\n{}\n{}".format(response1,
                                        response2,
                                        response3,
                                        response4)

    #write the responses to your workspace
    with open(os.path.join(model_path, api_return_file_name), "w") as f2:
        f2.write(responses)


if __name__ == "__main__":
    main()



