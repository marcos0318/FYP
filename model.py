import tensorflow as tf
from data import Data


class Model:
    def __init__(self, flags):
        with tf.variable_scope("model"):
            embedding_size = self.embedding_size = flags.embedding_size

            self.encoder_input = tf.placeholder(tf.float32, [None, None, None])
            self.decoder_input = tf.placeholder(tf.float32, [None, None, None])
            self.decoder_label = tf.placeholder(tf.float32, [None, None, None])

            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size)
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size)

            # the encoding steps
            initial_state = encoder_cell.zero_state(batch_size, dtype=tf.float32)
            _, self.encoded_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_input, dtype=tf.float32, initial_state = initial_state)

            # decoding steps
            self.decoder_output, _ = tf.nn.dynamic_rnn(decoder_cell, self.decoder_input, dtype=tf.float32, initial_state = self.encoded_state)

            self.loss = tf.reduce_sum((self.decoder_label - self.decoder_output) * (self.decoder_label - self.decoder_output))

            self.optimizer = tf.train.AdamOptimizer.minimize(self.loss)

            




