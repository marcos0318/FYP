import json
import numpy as np


class Data:
    def __init__(self, path):
        with open(path, "r") as fin:
            self.rawDict = json.load(fin)

        self.id2key = [int(key) for key, value in self.rawDict.items()]
        self.key2id = {key: i for i, key in enumerate(self.id2key)}

        self.Xs = np.array([value for key, value in self.rawDict.items()])
       
    def get_batch(self, batch_size = 3):
        """
        get_batch will return a tuple. The first is input for encoder and the output of the decoder (with EOS at the end). 
        the second is the input of decoder (with SOS in the front)
        """


        indices = np.random.randint(self.Xs.shape[0], size=batch_size)
        encoder_input = self.Xs[indices]
        decoder_input = np.zeros((encoder_input.shape[0], encoder_input.shape[1]+1, encoder_input.shape[2]))
        decoder_input[:, :-1, :] = encoder_input
        decoder_label = np.zeros((encoder_input.shape[0], encoder_input.shape[1]+1, encoder_input.shape[2]))
        decoder_label[:, 1:, :] = encoder_input

        return encoder_input, decoder_input, decoder_label




        

 
        
        
if __name__ == "__main__":
    data = Data("2016-06-01-2017-06-01all-factors.json")
    print(data.get_batch())