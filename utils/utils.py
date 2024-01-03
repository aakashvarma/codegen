import json
import re


def load_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def parse_json(file_path):
    data = load_from_json(file_path)
    return data


def parse_prompt(file_path):
    with open(file_path, "r") as file:
        eval_prompt = file.read()
    additional_info = ("You are a powerful text-to-SQL model. Your job is to answer questions about a database. You "
                       "are given a question and context regarding one or more tables.You must output the SQL query "
                       "that answers the question.")
    eval_prompt = additional_info + eval_prompt
    return eval_prompt


def str_or_bool(value):
    if str(value).lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif str(value).lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        return str(value)  # if it's not a recognizable boolean, treat it as a string

def calculate_accuracy(sql_output_arr, real_output_arr):
    correct_predictions = sum(1 for sql_output, real_output in zip(sql_output_arr, real_output_arr) if sql_output == real_output)
    total_predictions = len(sql_output_arr)
    accuracy = correct_predictions / total_predictions
    return accuracy

def extract_sql_output(sql_input):
    match = re.search(r'### Response:\s*(.*?)\s*(?:###|$)', sql_input, re.DOTALL)
    sql_query = match.group(1).strip()
    sql_output = re.sub(r'\n\s*\n', '\n', sql_query)
    return sql_output

def extract_question_context(text):
    context_pattern = re.compile(r'Context: (.+)', re.IGNORECASE)
    question_pattern = re.compile(r'Input: (.+)', re.IGNORECASE)

    context_match = context_pattern.search(text)
    question_match = question_pattern.search(text)

    context = context_match.group(1).strip() if context_match else None
    question = question_match.group(1).strip() if question_match else None

    return question, context
