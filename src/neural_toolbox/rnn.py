import tensorflow as tf
import tensorflow_addons as tfa


# For some reason, it is faster than MultiCell on tf
def variable_length_LSTM(inp, num_hidden, seq_length,
                         dropout_keep_prob=1.0, scope="lstm", depth=1,
                         layer_norm=False, reuse=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        states = []
        last_states = []
        rnn_states = inp
        for d in range(depth):
            with tf.compat.v1.variable_scope('lstmcell'+str(d)):

                cell = tfa.rnn.LayerNormLSTMCell(
                    num_hidden, dropout=1-dropout_keep_prob)

                rnn_states, rnn_last_states = tf.compat.v1.nn.dynamic_rnn(
                    cell,
                    rnn_states,
                    dtype=tf.float32,
                    sequence_length=seq_length,
                )
                print(rnn_last_states)
                print(rnn_states)
                states.append(rnn_states)
                last_states.append(rnn_last_states[1])

        states = tf.concat(states, axis=2)
        last_states = tf.concat(last_states, axis=1)

        return last_states, states

