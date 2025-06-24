import numpy as np
import os


def main():
    ex_num = 42
    example_file = f"output_{ex_num}.txt"
    file_1 = f"outputs_single/{example_file}"
    file_2 = f"outputs_double/{example_file}"

    output_1 = np.load(file_1)
    output_2 = np.load(file_2)

    print(f"Output 1: {output_1}")  
    print(f"Output 2: {output_2}")

    # check if the outputs are equal
    if np.array_equal(output_1, output_2):
        print("The outputs are equal.")
    else:
        print("The outputs are not equal.")

    #largest output
    max_1 = np.max(output_1)
    #min output
    min_1 = np.min(output_1)
    print(f"Max output 1: {max_1}")
    print(f"Min output 1: {min_1}")

    exit()

    
    # load labels
    example_file = f"labels_{ex_num}.txt"
    file_1 = f"outputs_single/{example_file}"
    file_2 = f"outputs_double/{example_file}"
    labels_1 = np.load(file_1)
    labels_2 = np.load(file_2)
    print(f"Labels 1: {labels_1}")
    print(f"Labels 2: {labels_2}")
    # check if the labels are equal
    if np.array_equal(labels_1, labels_2):
        print("The labels are equal.")
    else:
        print("The labels are not equal.")


if __name__ == "__main__":
    main()