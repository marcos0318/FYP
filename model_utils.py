import tensorflow as tf
from data import Data
import argparse
from tqdm import tqdm 
import os
import numpy as np
import json
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--latent_size", type = int, default = 100)
parser.add_argument("-b", "--batch_size", type = int, default = 64)
parser.add_argument("-e", "--num_epoch", type = int, default = 40)
parser.add_argument("-u", "--input_size", type = int, default = input_size)
parser.add_argument("-g", "--GPU", type = str, default = "0")
args = parser.parse_args()
print(args)


m = Model(args)

sess = tf.Session()

m.saver.restore(sess, "./saved_models/lstm_autoencoder.ckpt")

def lstm_encode(data):
    feed_dict = {
        m.encoder_input: encoder_input
    }

    c, h= sess.run(m.encoded_state, feed_dict = feed_dict)

    encoded_state = np.hstack((c, h))

    return encoded_state



encoder_input, decoder_input, decoder_label = data.get_all()

encoded_state = lstm_encode(encoder_input)


print(encoded_state.shape)