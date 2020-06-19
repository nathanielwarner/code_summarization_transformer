import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import json
import tqdm
import os
import argparse

import text_data_utils as tdu
from tokenizer import Tokenizer
from tf_utils import beam_search_decode, beam_search_decode_new, dataset_to_batched_tensors


# These functions and classes were originally created from the TensorFlow Transformer tutorial
# It has been refactored, and given the ability to create Universal Transformers,
# and those with shared queries and keys in the attention blocks.


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, shared_qk=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.shared_qk = shared_qk

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        if not self.shared_qk:
            self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        if self.shared_qk:
            k = self.wq(k)
        else:
            k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, shared_qk=False):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, shared_qk=shared_qk)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, shared_qk=False):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, shared_qk=shared_qk)
        self.mha2 = MultiHeadAttention(d_model, num_heads, shared_qk=False)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, universal=False, shared_qk=False):
        super(Encoder, self).__init__()

        self.universal = universal

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        if self.universal:
            self.enc_layer = EncoderLayer(d_model, num_heads, dff, rate, shared_qk=shared_qk)
        else:
            self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, shared_qk=shared_qk)
                               for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        if not self.universal:
            x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            if self.universal:
                x += self.pos_encoding[:, :seq_len, :]
                x += self.pos_encoding[:, i, :]  # Timestep encoding
                x = self.enc_layer(x, training, mask)
            else:
                x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1, universal=False, shared_qk=False):
        super(Decoder, self).__init__()

        self.universal = universal

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        if self.universal:
            self.dec_layer = DecoderLayer(d_model, num_heads, dff, rate, shared_qk=shared_qk)
        else:
            self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate, shared_qk=shared_qk)
                               for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = []

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        if not self.universal:
            x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            if self.universal:
                x += self.pos_encoding[:, :seq_len, :]
                x += self.pos_encoding[:, i, :]  # Timestep encoding
                x, block1, block2 = self.dec_layer(x, enc_output, training, look_ahead_mask, padding_mask)
            else:
                x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                       look_ahead_mask, padding_mask)

            attention_weights.append((block1, block2))

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


preprocessors = {
    'java': tdu.preprocess_java,
    'javadoc': tdu.preprocess_javadoc
}


class Transformer(tf.keras.Model):
    def __init__(self, model_path, train_set=None, val_set=None, num_train_epochs=0, train_batch_size=64,
                 sets_preprocessed=False):
        super(Transformer, self).__init__()

        model_path = os.path.abspath(model_path)
        with open(os.path.join(model_path, "transformer_description.json")) as transformer_desc_json:
            transformer_description = json.load(transformer_desc_json)

        self.output_tokenizer = Tokenizer(transformer_description['tar_tokenizer_type'],
                                          os.path.join(model_path, transformer_description['tar_tokenizer_path']),
                                          training_texts=([ex[0] for ex in train_set] if train_set is not None
                                                          else None),
                                          target_vocab_size=transformer_description['tar_target_vocab_size'])
        self.input_tokenizer = Tokenizer(transformer_description['inp_tokenizer_type'],
                                         os.path.join(model_path, transformer_description['inp_tokenizer_path']),
                                         training_texts=([ex[1] for ex in train_set] if train_set is not None
                                                         else None),
                                         target_vocab_size=transformer_description['inp_target_vocab_size'])

        self.tar_prep = preprocessors[transformer_description['tar_type']]
        self.inp_prep = preprocessors[transformer_description['inp_type']]

        num_layers = transformer_description['num_layers']
        d_model = transformer_description['d_model']
        dff = transformer_description['dff']
        num_heads = transformer_description['num_heads']

        dropout_rate = transformer_description['dropout_rate']

        universal = transformer_description['universal']
        shared_qk = transformer_description['shared_qk']

        self.max_input_len = transformer_description['inp_dim']
        self.max_output_len = transformer_description['tar_dim']

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               self.input_tokenizer.vocab_size, self.max_input_len, rate=dropout_rate,
                               universal=universal, shared_qk=shared_qk)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               self.output_tokenizer.vocab_size, self.max_output_len, rate=dropout_rate,
                               universal=universal, shared_qk=shared_qk)

        self.final_layer = tf.keras.layers.Dense(self.output_tokenizer.vocab_size)

        learning_rate = CustomSchedule(d_model)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                                  epsilon=1e-9)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        checkpoint_path = os.path.join(model_path, "train")

        ckpt = tf.train.Checkpoint(transformer=self,
                                   optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        if num_train_epochs > 0:
            self.train(train_set, val_set, batch_size=train_batch_size, num_epochs=num_train_epochs,
                       sets_preprocessed=sets_preprocessed)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    @tf.function
    def train_step(self, inp, tar_inp, tar_out):

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.call(inp, tar_inp,
                                       True,
                                       enc_padding_mask,
                                       combined_mask,
                                       dec_padding_mask)
            loss = loss_function(tar_out, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_out, predictions)

    def val_loss(self, dataset, batch_size=64):
        dataset, num_batches = dataset_to_batched_tensors(dataset, batch_size, self.max_output_len, self.max_input_len)
        loss = 0.0
        batch_nums = tqdm.trange(num_batches)
        for i in batch_nums:
            tar, inp = next(dataset)
            tar_inp = tar[:, :-1]
            tar_out = tar[:, 1:]
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
            predictions, _ = self.call(inp, tar_inp,
                                       False,
                                       enc_padding_mask,
                                       combined_mask,
                                       dec_padding_mask)
            loss += loss_function(tar_out, predictions)
            if (i + 1) % 10 == 0:
                current_loss = loss / (i + 1)
                batch_nums.set_description("evaluate: loss=%.4f" % current_loss)
        loss /= num_batches
        return loss

    def train(self, train_set, val_set, batch_size=64, num_epochs=100, sets_preprocessed=False):

        if not sets_preprocessed:
            print("Preprocessing datasets...")
            train_set = [(self.tar_prep(s), self.inp_prep(c)) for s, c in train_set]
            val_set = [(self.tar_prep(s), self.inp_prep(c)) for s, c in val_set]

        print("Tokenizing datasets...")
        train_set = [(self.output_tokenizer.tokenize_text(s), self.input_tokenizer.tokenize_text(c))
                     for s, c in train_set]
        val_set = [(self.output_tokenizer.tokenize_text(s), self.input_tokenizer.tokenize_text(c))
                   for s, c in val_set]

        print("Removing examples that are too long...")
        train_set = [(s, c) for s, c in train_set if len(s) <= self.max_output_len and len(c) <= self.max_input_len]
        val_set = [(s, c) for s, c in val_set if len(s) <= self.max_output_len and len(c) <= self.max_input_len]

        print("Training on %d examples, validating on %d examples" % (len(train_set), len(val_set)))

        best_val_loss = self.val_loss(val_set, batch_size=batch_size)
        print('Initial Validation loss: {:.4f}'.format(best_val_loss))
        num_epochs_with_no_improvement = 0

        for epoch in range(num_epochs):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            train_batches, num_batches = dataset_to_batched_tensors(train_set, batch_size,
                                                                    self.max_output_len, self.max_input_len)
            batch_nums = tqdm.trange(num_batches)
            for batch_num in batch_nums:
                tar, inp = next(train_batches)
                tar_inp = tar[:, :-1]
                tar_out = tar[:, 1:]
                self.train_step(inp, tar_inp, tar_out)
                if batch_num % 50 == 0:
                    batch_nums.set_description("Epoch {} of {}, Loss {:.4f}, Accuracy {:.4f}".format(
                        epoch + 1, num_epochs, self.train_loss.result(), self.train_accuracy.result()))

            val_loss = self.val_loss(val_set, batch_size=batch_size)
            print('Validation loss: {:.4f}'.format(val_loss))

            if val_loss < best_val_loss:
                num_epochs_with_no_improvement = 0
                ckpt_save_path = self.ckpt_manager.save()
                best_val_loss = val_loss
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
            else:
                num_epochs_with_no_improvement += 1
                print("Val loss did not improve")
                if num_epochs_with_no_improvement > 8:
                    print("Early stopping")
                    break

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                self.train_loss.result(),
                                                                self.train_accuracy.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def _single_bsd_step(self, predicted_so_far, state):
        enc_input = state["enc_input"]
        enc_output = state["enc_output"]
        tar = tf.expand_dims(predicted_so_far, 0)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            enc_input, tar)
        # dec_output.shape == (1, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, False, combined_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # (1, tar_seq_len, target_vocab_size)
        state["attention_weights"] = attention_weights
        return tf.nn.softmax(final_output[0][-1]), state

    def evaluate_on_sentence(self, inp_sentence, max_length):

        encoder_input = self.input_tokenizer.tokenize_text(inp_sentence)
        if len(encoder_input) > self.max_input_len:
            print("Warning: Input sentence exceeds maximum length")
        encoder_input = tf.keras.preprocessing.sequence.pad_sequences([encoder_input], maxlen=self.max_input_len,
                                                                      dtype='int32', padding='post', value=0,
                                                                      truncating='post')

        enc_padding_mask = create_padding_mask(encoder_input)
        enc_output = self.encoder(encoder_input, False, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        dec_state = {
            "enc_input": encoder_input,
            "enc_output": enc_output
        }

        best_beam = beam_search_decode(dec_state, self._single_bsd_step, self.output_tokenizer.start_token,
                                       self.output_tokenizer.end_token, beam_width=1, max_len=max_length)

        return best_beam[0], best_beam[2]["attention_weights"]

    def plot_attention_weights(self, attention, sentence, result):

        sentence = self.input_tokenizer.tokenize_text(sentence)
        sentence = tf.keras.preprocessing.sequence.pad_sequences([sentence], maxlen=self.max_input_len,
                                                                 dtype='int32', padding='post', value=0,
                                                                 truncating='post')[0]
        inp_mask = tf.logical_not(tf.equal(sentence, 0))
        sentence_ragged = tf.ragged.boolean_mask(sentence, inp_mask)
        if not tf.is_tensor(sentence_ragged):
            sentence = sentence_ragged.to_tensor()
        else:
            sentence = sentence_ragged
        result_no_sos = result[1:]

        encoder_decoder_attention = tf.squeeze(tf.convert_to_tensor([att_layer[1] for att_layer in attention]), axis=1)

        total_attention = tf.reduce_sum(encoder_decoder_attention[-1], axis=0)
        zeros_mask = tf.logical_not(tf.equal(total_attention, 0))
        total_attention_ragged = tf.ragged.boolean_mask(total_attention, zeros_mask)
        total_attention_non_ragged = total_attention_ragged.to_tensor()
        total_attention_softmax = tf.nn.softmax(total_attention_non_ragged, axis=-1)

        fig = plt.figure(figsize=(6, 3), dpi=192)
        ax = fig.add_subplot(1, 1, 1)

        # plot the attention weights
        ax.matshow(total_attention_softmax, cmap='viridis')

        fontdict = {'fontsize': 6}

        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(result_no_sos)))

        ax.set_xticklabels(
            [self.input_tokenizer.de_tokenize_text([i]) for i in sentence],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([self.output_tokenizer.de_tokenize_text([i]) for i in result_no_sos],
                           fontdict=fontdict)

        ax.set_xlabel('Encoder-Decoder Attention')

        plt.show()

    def translate(self, sentence, plot=False, print_output=False):

        sentence = self.inp_prep(sentence)
        result, attention_weights = self.evaluate_on_sentence(sentence, self.max_output_len)

        predicted_sentence = tdu.de_eof_text(self.output_tokenizer.de_tokenize_text(result))

        if print_output:
            print('Input: {}'.format(sentence))
            print('Predicted translation: {}'.format(predicted_sentence))

        if plot:
            self.plot_attention_weights(attention_weights, sentence, result)

        return predicted_sentence

    def translate_batch(self, sentences, preprocessed=False, beam_width=1):
        num_examples = len(sentences)
        if not preprocessed:
            sentences = map(self.inp_prep, sentences)
        encoder_inputs = list(map(self.input_tokenizer.tokenize_text, sentences))
        encoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(encoder_inputs, maxlen=self.max_input_len,
                                                                       dtype='int32', padding='post', value=0,
                                                                       truncating='post')

        enc_padding_mask = create_padding_mask(encoder_inputs)
        enc_outputs = self.encoder(encoder_inputs, False, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        expanded_enc_inputs = tf.repeat(tf.expand_dims(encoder_inputs, 1), beam_width, axis=1)
        expanded_enc_outputs = tf.repeat(tf.expand_dims(enc_outputs, 1), beam_width, axis=1)
        flat_enc_inputs = tf.reshape(expanded_enc_inputs, (num_examples * beam_width,
                                                           self.max_input_len))
        flat_enc_outputs = tf.reshape(expanded_enc_outputs, (num_examples * beam_width,
                                                             self.max_input_len, -1))

        def single_bsd_step(tar, state):
            _, combined_mask, dec_padding_mask = create_masks(flat_enc_inputs, tar)
            # dec_output.shape == (batch_size, tar_seq_len, d_model)
            dec_output, attention_weights = self.decoder(tar, flat_enc_outputs, False, combined_mask,
                                                         dec_padding_mask)
            final_output = self.final_layer(dec_output)
            really_final = tf.nn.softmax(final_output[:, -1, :])
            return really_final, state

        nothing = tf.zeros(tf.shape(sentences)[0], 1)
        dec_outputs = beam_search_decode_new(nothing, single_bsd_step, self.output_tokenizer.start_token,
                                             self.output_tokenizer.end_token, beam_width=beam_width,
                                             max_len=self.max_output_len)

        de_tokenized = map(self.output_tokenizer.de_tokenize_text, dec_outputs)
        final = map(tdu.de_eof_text, de_tokenized)

        return final

    def interactive_demo(self):
        while True:
            print()
            inp = input(">> ")
            if inp == "exit":
                break
            self.translate(inp, plot=True, print_output=True)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def main():
    parser = argparse.ArgumentParser(description="Demonstrate the translation abilities of the Transformer")
    parser.add_argument("--model_path", help="Path to the Transformer model", required=True)
    args = vars(parser.parse_args())
    model_path = args["model_path"]

    transformer = Transformer(model_path)
    transformer.interactive_demo()


if __name__ == "__main__":
    main()
