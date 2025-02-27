import os
import numpy as np

def concatenate_and_delete(folder_a, folder_b):
    """
    Concatenates corresponding .npy files from folder A and folder B,
    saving the concatenated files in folder A and deleting the files from folder B.

    Args:
        folder_a: Path to folder A.
        folder_b: Path to folder B.
    """

    for filename in os.listdir(folder_a):
        if filename.endswith(".npy"):  # Process only .npy files
            filepath_a = os.path.join(folder_a, filename)
            filepath_b = os.path.join(folder_b, filename)

            if os.path.exists(filepath_b):  # Check if corresponding file exists in folder B
                try:
                    data_a = np.load(filepath_a)
                    data_b = np.load(filepath_b)

                    # Determine the concatenation axis (adjust if needed)
                    axis = 0  # Concatenate along the first axis (rows) - common for stacking data

                    concatenated_data = np.concatenate((data_a, data_b), axis=axis)

                    np.save(filepath_a, concatenated_data)  # Save concatenated data back to folder A, overwriting the original
                    os.remove(filepath_b)  # Delete the file from folder B
                    print(f"Concatenated and deleted: {filename}")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

            else:
                print(f"Warning: Corresponding file not found in folder B for {filename}")


# Example usage:
folder_a = "A"
folder_b = "B"

concatenate_and_delete(folder_a, folder_b)

print("Finished processing.")
