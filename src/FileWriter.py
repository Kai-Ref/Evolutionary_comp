import numpy as np
class FileWriter:
    def __init__(self, file_path: str = "data"):
        self.file_path = file_path
        
    def __call__(self, data: np.ndarray, name: str) -> None:
        np.save(f"{self.file_path}/{name}.npy", data)
        print(f"Data saved to {name}_{self.file_path}")