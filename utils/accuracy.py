import pickle
import argparse
import re


def calculate_accuracy(sql_output_arr, real_output_arr):
    mismatched_indices = [i for i, (sql_output, real_output) in enumerate(zip(sql_output_arr, real_output_arr)) if
                          sql_output != real_output]

    print("Mismatched predictions:")
    for index in mismatched_indices:
        print(f"Index {index}: SQL Output: {sql_output_arr[index]}, Real Output: {real_output_arr[index]}")
        print()

    correct_predictions = len(sql_output_arr) - len(mismatched_indices)
    total_predictions = len(sql_output_arr)
    accuracy = correct_predictions / total_predictions
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Calculate accuracy from pickled data.')
    parser.add_argument('file_path', type=str, help='Path to the pickled file')

    args = parser.parse_args()

    with open(args.file_path, 'rb') as file:
        loaded_sql_output_arr, loaded_real_output_arr = pickle.load(file)

    sql_output_arr = []
    real_output_arr = []
    for batch_sql_output, batch_real_output in zip(loaded_sql_output_arr, loaded_real_output_arr):
        for sql_output, real_output in zip(batch_sql_output, batch_real_output):
            match = re.search(r'### Response:\n(.+)', sql_output, re.DOTALL)
            sql_query = match.group(1).strip()
            sql_output = re.sub(r'\n\s*\n', '\n', sql_query)
            sql_output_arr.append(sql_output)
            real_output_arr.append(real_output)

    accuracy = calculate_accuracy(sql_output_arr, real_output_arr)

    print(f'Accuracy: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    main()
