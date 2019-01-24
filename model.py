import tensorflow as tf
from data import Data
import argparse
from tqdm import tqdm 

class Model:
    def __init__(self, flags):
        with tf.variable_scope("model"):
            embedding_size = self.embedding_size = flags.latent_size

            self.encoder_input = tf.placeholder(tf.float32, [flags.batch_size, None, flags.input_size])
            self.decoder_input = tf.placeholder(tf.float32, [flags.batch_size, None, flags.input_size])
            self.decoder_label = tf.placeholder(tf.float32, [flags.batch_size, None, flags.input_size])

            encoder_cell = tf.nn.rnn_cell.LSTMCell(embedding_size)
            decoder_cell = tf.nn.rnn_cell.LSTMCell(embedding_size)

            # the encoding steps
            initial_state = encoder_cell.zero_state(flags.batch_size, dtype=tf.float32)
            with tf.variable_scope("encoder"):
                _, self.encoded_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_input, dtype=tf.float32, initial_state = initial_state)

            # decoding steps
            with tf.variable_scope("decoder"):
                self.decoder_output, _ = tf.nn.dynamic_rnn(decoder_cell, self.decoder_input, dtype=tf.float32, initial_state = self.encoded_state)
            
            self.loss = tf.reduce_sum((self.decoder_label - self.decoder_output) * (self.decoder_label - self.decoder_output))
            self.optimizer = tf.train.AdamOptimizer.minimize(self.loss)



if __name__ == "__main__":
    data = Data("2016-06-01-2017-06-01all-factors.json")
    input_size = data.Xs.shape[-1]

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--latent_size", type = int, default = 100)
    parser.add_argument("-b", "--batch_size", type = int, default = 64)
    parser.add_argument("-e", "--num_epoch", type = int, default = 10)
    parser.add_argument("-u", "--input_size", type = int, default = input_size)
    args = parser.parse_args()
    print(args)


   
    model = Model(args)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Model Initialized")

        for i in range(args.num_epoch):
            print("Epoch ", i+1, "/", args.num_epoch)

            num_batches = 5000 // args.batch_size

            process_bar = tqdm(range(num_batches))
            for i in process_bar:
                encoder_input, decoder_input, decoder_label = data.get_batch()

                feed_dict = {
                    model.encoder_input: encoder_input,
                    model.decoder_input: decoder_input,
                    model.decoder_label: decoder_label
                }

                loss, _ = sess.run([model.loss, model.optimizer], feed_dict = feed_dict)
                process_bar.set_description("Loss: %0.2f" % loss)



