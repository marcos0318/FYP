from data import Data
import argparse
from tqdm import tqdm 
import os
import numpy as np
import json
from model import Model
from model_utils import lstm_encode

file_name = "2016-06-01-2017-06-01all-factors.json"



"factor input 10"
"sector input 29"

data = Data(file_name)






encoder_input, decoder_input, decoder_label = data.get_all()






encoded_state = lstm_encode(encoder_input)


print(encoded_state.shape)