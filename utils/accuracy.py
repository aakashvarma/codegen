import pickle
import argparse

def calculate_accuracy(sql_output_arr, real_output_arr):
    correct_predictions = sum(1 for sql_output, real_output in zip(sql_output_arr, real_output_arr) if sql_output == real_output)
    total_predictions = len(sql_output_arr)
    accuracy = correct_predictions / total_predictions
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Calculate accuracy from pickled data.')
    parser.add_argument('file_path', type=str, help='Path to the pickled file')

    args = parser.parse_args()

    with open(args.file_path, 'rb') as file:
        loaded_sql_output_arr, loaded_real_output_arr = pickle.load(file)

    accuracy = calculate_accuracy(loaded_sql_output_arr, loaded_real_output_arr)

    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
