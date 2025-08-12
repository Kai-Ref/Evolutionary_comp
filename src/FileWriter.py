class FileWriter:
    def __init__(self, file_path: str = "output.txt"):
        self.file_path = file_path
        
    def __call__(self):
        raise NotImplementedError("FileWriter not yet implemented")