import json
import sys


def load_from_json(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded data.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def parse_json(file_path):
    """
    Parse data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The parsed data.
    """
    data = load_from_json(file_path)
    return data


def parse_prompt(file_path):
    """
    Parse text from a file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The parsed text.
    """
    with open(file_path, "r") as file:
        eval_prompt = file.read()
    additional_info = "You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.You must output the SQL query that answers the question. "
    eval_prompt = additional_info + eval_prompt
    return eval_prompt


def str_or_bool(value):
    """
    Convert a string representation of a boolean to a boolean or return the input as a string.

    Args:
        value (str): The input value.

    Returns:
        Union[bool, str]: The converted boolean or the original string.
    """
    if str(value).lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif str(value).lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        return str(value)  # if it's not a recognizable boolean, treat it as a string
