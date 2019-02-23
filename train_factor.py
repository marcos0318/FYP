from model import Model
import tensorflow as tf
from data import Data
import argparse
from tqdm import tqdm 
import os
import numpy as np
import json



file_name = "2016-06-01-2017-06-01all-factors.json"

data = Data(file_name)
print(data.Xs.shape)
input_size = data.Xs.shape[-1]

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--latent_size", type = int, default = 100)
parser.add_argument("-b", "--batch_size", type = int, default = 64)
parser.add_argument("-e", "--num_epoch", type = int, default = 40)
parser.add_argument("-u", "--input_size", type = int, default = input_size)
parser.add_argument("-g", "--GPU", type = str, default = "0")
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU

model = Model(args)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Model Initialized")

    for i in range(args.num_epoch):
        print("Epoch ", i+1, "/", args.num_epoch)

        num_batches = 5000 // args.batch_size

        process_bar = tqdm(range(num_batches))
        for i in process_bar:
            encoder_input, decoder_input, decoder_label = data.get_batch(args.batch_size)

            feed_dict = {
                model.encoder_input: encoder_input,
                model.decoder_input: decoder_input,
                model.decoder_label: decoder_label
            }

            loss, _ = sess.run([model.loss, model.optimizer], feed_dict = feed_dict)
            process_bar.set_description("Loss: %0.2f" % loss)

        model.saver.save(sess, "./saved_models/lstm_autoencoder_factor.ckpt")

    # encoder_input, decoder_input, decoder_label = data.get_all()

    # feed_dict = {
    #     model.encoder_input: encoder_input
    # }

    # c, h= sess.run(model.encoded_state, feed_dict = feed_dict)

    # encoded_state = np.hstack((c, h))

    # print(encoded_state.shape)

    # features = { data.id2key[i]: encoded_state[i, :].tolist() for i in range(encoded_state.shape[0])}

    # with open("encoded_" + file_name, "w") as fout: