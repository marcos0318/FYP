import tensorflow as tf
from data import Data
import argparse
from tqdm import tqdm 
import os
import numpy as np
import json
from model import Model
from sklearn.manifold import TSNE


"factor input 10"
"sector input 29"


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--latent_size", type = int, default = 100)
parser.add_argument("-b", "--batch_size", type = int, default = 64)
parser.add_argument("-e", "--num_epoch", type = int, default = 40)
parser.add_argument("-u", "--input_size", type = int, default = 10)
parser.add_argument("-g", "--GPU", type = str, default = "0")
args = parser.parse_args()
print(args)


factor_graph = tf.Graph()
with factor_graph.as_default():
    factor_model = Model(args)

args.input_size = 29
print(args)

sector_graph = tf.Graph()
with sector_graph.as_default():
    sector_model = Model(args)


factor_sess = tf.Session(graph=factor_graph)
sector_sess = tf.Session(graph=sector_graph)



with factor_sess.as_default():
    with factor_graph.as_default():
        tf.global_variables_initializer().run()
        factor_model.saver.restore(factor_sess, "./saved_models/lstm_autoencoder_factor.ckpt")

with sector_sess.as_default():
    with sector_graph.as_default():
        tf.global_variables_initializer().run()
        sector_model.saver.restore(sector_sess, "./saved_models/lstm_autoencoder_sector.ckpt")




def lstm_factor_encode(data):
    feed_dict = {
        factor_model.encoder_input: data
    }

    c, h= factor_sess.run(factor_model.encoded_state, feed_dict = feed_dict)

    encoded_state = np.hstack((c, h))
    return encoded_state

def lstm_sector_encode(data):
    feed_dict = {
        sector_model.encoder_input: data
    }

    c, h= sector_sess.run(sector_model.encoded_state, feed_dict = feed_dict)

    encoded_state = np.hstack((c, h))
    return encoded_state


def lstm_tsne_factor_encode(data):
    encoded_data = lstm_factor_encode(data)
    encoded_data = TSNE(n_components=2).fit_transform(encoded_data)
    return encoded_data

def lstm_tsne_sector_encode(data):
    encoded_data = lstm_sector_encode(data)
    encoded_data = TSNE(n_components=2).fit_transform(encoded_data)
    return encoded_data