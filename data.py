import json
import numpy as np


class Data:
    def __init__(self, path):
        with open(path, "r") as fin:
            self.rawDict = json.load(fin)

        self.id2key = [int(key) for key, value in self.rawDict.items()]
        self.key2id = {key: i for i, key in enumerate(self.id2key)}

        self.Xs = np.array([value for key, value in self.rawDict.items()])
        print(self.id2key)
        print(self.key2id)
        print(self.Xs)

    def get_batch():
        """
        get_batch will return a tuple. The first one is the get batch. The second one is the 
        """
        pass


        
        
if __name__ == "__main__":
    data = Data("2016-06-01-2017-06-01all-factors.json")