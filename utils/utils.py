import json
import sys

def load_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def parse_json(file_path):
    data = load_data_from_json(file_path)
    return data

def parse_text(file_path):
    with open(file_path, 'r') as file:
        eval_prompt = file.read()
    return eval_prompt

def str_or_bool(value):
    if str(value).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(value).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return str(value)  # if it's not a recognizable boolean, treat it as a string