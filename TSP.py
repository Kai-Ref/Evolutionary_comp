class TSPFileReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.metadata = {}
        self.node_coords = {}

    def read(self):
        with open(self.filepath, 'r') as file:
            lines = file.readlines()

        in_node_section = False
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line == "NODE_COORD_SECTION":
                in_node_section = True
                continue
            elif line == "EOF":
                break

            if in_node_section:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    self.node_coords[node_id] = (x, y)
            else:
                if ':' in line:
                    key, value = line.split(':', 1)
                    self.metadata[key.strip()] = value.strip()

    def get_metadata(self):
        return self.metadata

    def get_node_coords(self):
        return self.node_coords

    def __repr__(self):
        return f"<TSPFileReader {self.filepath}, {len(self.node_coords)} nodes>"




if __name__ == "__main__":
    reader = TSPFileReader('C:/Users/555ka/Coding/Evolutionary_comp/datasets/eil51.tsp')
    reader.read()

    print(reader.get_metadata())
    print(len(reader.get_node_coords()))
    print(reader.get_node_coords()[1]) 