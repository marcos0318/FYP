import json
import numpy as np


class Data:
    def __init__(self, path):
        with open(path, "r") as fin:
            self.rawDict = json.load(fin)

        self.id2key = [int(key) for key, value in self.rawDict.items()]
        self.key2id = {key: i for i, key in enumerate(self.id2key)}

        self.Xs = np.array([value for key, value in self.rawDict.items()])
       
    def get_batch():
        """
        get_batch will return a tuple. The first is input for encoder and the output of the decoder (with EOS at the end). 
        the second is the input of decoder (with SOS in the front)
        """

        indices = np.random.randint(Xs.shape[0], size=batch_size)
        return self.Xs[indices]

        


        
        
if __name__ == "__main__":
    data = Data("2016-06-01-2017-06-01all-factors.json")