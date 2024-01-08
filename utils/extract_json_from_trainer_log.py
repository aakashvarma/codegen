# import re
# import ast
# import argparse
#
# def extract_json_data_from_text(text):
#     # Define a regular expression to extract JSON data
#     json_pattern = re.compile(r'\{.*?\}')
#
#     # Initialize a list to store extracted JSON data
#     json_data_list = []
#
#     # Find all matches for JSON data
#     matches = json_pattern.finditer(text)
#     for match in matches:
#         try:
#             json_data = ast.literal_eval(match.group())
#             json_data_list.append(json_data)
#         except (SyntaxError, ValueError) as e:
#             print(f"Error evaluating JSON-like string: {e}")
#             print(f"Problematic string: {match.group()}")
#
#     return json_data_list
#
# def collect_values_by_key(json_data_list, keys_to_collect):
#     # Initialize a dictionary to store values by key
#     values_by_key = {key: [] for key in keys_to_collect}
#
#     # Use a set to keep track of unique epoch values
#     unique_epochs = set()
#
#     # Iterate through each JSON data in the list
#     for json_data in json_data_list:
#         for key, value in json_data.items():
#             if key in keys_to_collect:
#                 # Special handling for 'epoch' key to exclude duplicates
#                 if key == 'epoch' and value in unique_epochs:
#                     continue
#                 values_by_key[key].append(value)
#
#                 # Update the set for 'epoch'
#                 if key == 'epoch':
#                     unique_epochs.add(value)
#
#     return values_by_key
#
# def main():
#     # Create a command-line argument parser
#     parser = argparse.ArgumentParser(description="Extract JSON data from text file")
#     parser.add_argument("file_path", help="Path to the text file")
#
#     # Parse the command-line arguments
#     args = parser.parse_args()
#
#     # Read the content from the file
#     with open(args.file_path, 'r') as file:
#         text = file.read()
#
#     # Extract JSON data from the text
#     json_data_list = extract_json_data_from_text(text)
#
#     # Specify the keys to collect
#     keys_to_collect = ['loss', 'epoch', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'gen_len', 'eval_loss', 'eval_runtime', 'train_runtime', 'train_loss']
#
#     # Collect values by key
#     values_by_key = collect_values_by_key(json_data_list, keys_to_collect)
#
#     # Print the collected values by key
#     for key, values in values_by_key.items():
#         print(f"{key}: {values}")
#         print("\n")
#
# if __name__ == "__main__":
#     main()


import re
import ast
import argparse
import matplotlib.pyplot as plt

def extract_json_data_from_text(text):
    json_pattern = re.compile(r'\{.*?\}')
    json_data_list = []

    matches = json_pattern.finditer(text)
    for match in matches:
        try:
            json_data = ast.literal_eval(match.group())
            json_data_list.append(json_data)
        except (SyntaxError, ValueError) as e:
            print(f"Error evaluating JSON-like string: {e}")
            print(f"Problematic string: {match.group()}")

    return json_data_list

def collect_values_by_key(json_data_list, keys_to_collect):
    values_by_key = {key: [] for key in keys_to_collect}

    unique_epochs = set()

    for json_data in json_data_list:
        for key, value in json_data.items():
            if key in keys_to_collect:
                # Special handling for 'epoch' key to exclude duplicates
                if key == 'epoch' and value in unique_epochs:
                    continue
                values_by_key[key].append(value)

                if key == 'epoch':
                    unique_epochs.add(value)

    return values_by_key


def plot_graphs(values_by_key):
    # Plot separate graphs for each key except 'train_runtime' and 'train_loss'
    keys_to_plot = [key for key in values_by_key.keys() if key not in ['train_runtime', 'train_loss']]

    values_by_key['epoch'] = values_by_key['epoch'][:-1]
    for key in keys_to_plot:
        plt.figure()
        values = values_by_key[key]
        plt.plot(values_by_key['epoch'], values, marker='o', linestyle='-', color='b', label=key)
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.title(f'Graph for {key}')
        plt.grid(True)

        # Annotate each point with its epoch number
        for i, value in enumerate(values):
            plt.annotate(f'{values_by_key["epoch"][i]}', (values_by_key['epoch'][i], value), textcoords="offset points",
                         xytext=(0, 10), ha='center')

        plt.legend()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Extract JSON data from text file and plot graphs")
    parser.add_argument("file_path", help="Path to the text file")

    args = parser.parse_args()

    with open(args.file_path, 'r') as file:
        text = file.read()

    json_data_list = extract_json_data_from_text(text)

    keys_to_collect = ['loss', 'epoch', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'gen_len', 'eval_loss', 'eval_runtime', 'train_runtime', 'train_loss']
    values_by_key = collect_values_by_key(json_data_list, keys_to_collect)

    plot_graphs(values_by_key)

if __name__ == "__main__":
    main()
