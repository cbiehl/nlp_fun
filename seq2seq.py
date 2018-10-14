import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

class Encoder(object):
    def __init__(self, num_units=150):
        self.num_units = num_units
        self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    def forward(self, x, sl, reverse=True, final=True):
        # initialize with zeros
        state = self.encoder_cell.zero_state(len(x), dtype=tf.float32)
        timestep_x = tf.unstack(x, axis=1) #list of each timestep
        if reverse:
            timestep_x = reversed(timestep_x)
        outputs, cell_states = [], []
        for input_step in timestep_x:
            output, state = self.encoder_cell(input_step, state)
            outputs.append(output)
            cell_states.append(state[0])

        # format as tensor [batch_size, time_step, feature_dim]
        outputs = tf.stack(outputs, axis=1)
        cell_states = tf.stack(cell_states, axis=1)

        if final:
            if reverse:
                final_output = outputs[:, -1, :]
                final_cell_state = cell_states[:, -1, :]
            else:
                # get end index for each cell
                idx_last_output = tf.stack([tf.range(len(x)), sl], axis=1)
                # get last output for each sequence
                final_output = tf.gather_nd(outputs, idx_last_output)
                final_cell_state = tf.gather_nd(cell_states, idx_last_output)

            return final_output, final_cell_state

        else:
            return outputs, cell_states

    def save(self, path='Encoder/'):
        saver = tfe.Saver(self.encoder_cell.variables)
        saver.save(path)

    def load(self, path='Encoder/'):
        # do forward pass first to construct the graph again
        self.forward(np.zeros((32, 16, 300), dtype=np.float32), list(range(2, 34, 1)))
        saver = tfe.Saver(self.encoder_cell.variables)
        saver.restore(path)


class Decoder(object):
    def __init__(self, word2idx, idx2word, idx2emb, num_units=150, max_tokens=128):
        self.w2i = word2idx
        self.i2w = idx2word
        self.i2e = idx2emb
        self.num_units = num_units
        self.max_tokens = max_tokens
        self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        self.word_predictor = tf.layers.Dense(len(word2idx), activation=None)

    def forward(self, x, sos, state, training=False):
        output = tf.convert_to_tensor(sos, dtype=tf.float32)
        words_predicted, word_logits = [], []

        for mt in range(self.max_tokens):
            output, state = self.decoder_cell(output, state)
            logits = self.word_predictor(output) # [batch_size, vocab_size]
            logits = tf.nn.softmax(logits), state
            pred_word = tf.argmax(logits, 1).numpy()

            if training:
                output = x[:, mt, :]
            else:
                output = [self.i2e[i] for i in pred_word]

            words_predicted.append(pred_word)
            word_logits.append(logits)

        word_logits = tf.stack(word_logits, axis=1)

        # [max_tokens, num_samples] -> [num_samples, max_tokens]
        words_predicted = tf.stack(words_predicted, axis=1)

        return words_predicted, word_logits


def train():
    x, y, sl, sos, w2i, i2w, i2e = get_data()
    optimizer = tf.train.AdamOptimizer()
    encoder = Encoder()
    decoder = Decoder(w2i, i2w, i2e)

    for epoch in range(300):
        for x_batch, y_batch, sl_batch in zip(x, y, sl):
            optimizer.minimize(lambda: get_loss(encoder, decoder, x, y, sl, sos))


def cost_function(wl, y, sl):
    cross_entropy = y * tf.log(tf.clip_by_value(wl, 1e-10, 1.0))
    cross_entropy = - tf.reduce_sum(cross_entropy, 2)

    mask = tf.cast(tf.sequence_mask(sl, wl[1]), dtype=tf.float32)

    # Average over sequence lengths
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)

    return tf.reduce_mean(cross_entropy)


def get_loss(encoder, decoder, x, y, sl, sos):
    output, cell_state = encoder.forward(x, sl)
    _, wl = decoder.forward(x, sos, (cell_state, output), training=True)

    loss = cost_function(wl, y, sl)

    return loss