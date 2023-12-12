import json
import sys

def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def parse_config_json(file_path):
    data = load_data_from_json(file_path)
    return data

def parse_prompt_text(file_path):
    with open(file_path, 'r') as file:
        eval_prompt = file.read()
    return eval_prompt
