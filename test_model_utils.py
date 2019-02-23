from data import Data
import argparse
from tqdm import tqdm 
import os
import numpy as np
import json
from model import Model
from model_utils import lstm_factor_encode, lstm_sector_encode, lstm_tsne_factor_encode, lstm_tsne_sector_encode
from time 

"factor input 10"
"sector input 29"


file_name = "2016-06-01-2017-06-01all-factors.json"
factor_data = Data(file_name)
file_name = "2016-06-01-2017-06-01all-sectors.json"
sector_data = Data(file_name)





# Test factor data encoder
encoder_input, decoder_input, decoder_label = factor_data.get_all()
s = time.time()
print("Before:", encoder_input.shape)
encoded_state = lstm_factor_encode(encoder_input)
print("After:", encoded_state.shape, time.time()-s)

encoder_input, decoder_input, decoder_label = factor_data.get_all()
s = time.time()
print("Before:", encoder_input.shape)
encoded_state = lstm_tsne_factor_encode(encoder_input)
print("After:", encoded_state.shape, time.time()-s)

# Test sector data encoder


encoder_input, decoder_input, decoder_label = sector_data.get_all()
print("Before:", encoder_input.shape)
s = time.time()
encoded_state = lstm_sector_encode(encoder_input)
print("After:", encoded_state.shape, time.time()-s)


encoder_input, decoder_input, decoder_label = sector_data.get_all()
print("Before:", encoder_input.shape)
s = time.time()
encoded_state = lstm_tsne_sector_encode(encoder_input)
print("After:", encoded_state.shape, time.time()-s)

