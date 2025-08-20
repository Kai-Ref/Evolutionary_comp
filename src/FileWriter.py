import numpy as np
import os

class FileWriter:
    def __init__(self, file_path: str = "data"):
        self.file_path = file_path
        
    def __call__(self, data: np.ndarray, name: str) -> None:
        # Remove extension if present
        name_no_ext = os.path.splitext(name)[0]

        # Full path including base_path
        full_path = os.path.join(self.file_path, name_no_ext + ".npy")

        # Create all directories in the path if they don't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Save the file
        np.save(full_path, data)
        print(f"Data saved to {full_path}")