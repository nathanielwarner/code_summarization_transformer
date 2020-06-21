import tensorflow as tf
import sentencepiece as spm

import time
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
import sys

from data.leclair_java.process_funcom import preprocess_java, preprocess_javadoc


def beam_search_decode(initial_state, single_bsd_step, start_token, end_token, beam_width=10, max_len=50):
    final, initial_state = single_bsd_step(tf.expand_dims(start_token, 0), initial_state)
    predictions = tf.argsort(final, axis=-1, direction='DESCENDING').numpy()[0:beam_width]
    beams = []
    for k in range(beam_width):
        formed_candidate = ([start_token, predictions[k]],
                            -tf.math.log(final[predictions[k]]),
                            initial_state)
        beams.append(formed_candidate)
    for j in range(max_len - 1):
        candidates = []
        for k in range(beam_width):
            if beams[k][0][-1] == end_token:
                if len(candidates) < beam_width:
                    candidates.append(beams[k])
                else:
                    for m in range(len(candidates)):
                        if candidates[m][1] > beams[k][1]:
                            candidates[m] = beams[k]
                            break
            else:
                final, new_state = single_bsd_step(beams[k][0], beams[k][2])
                predictions = tf.argsort(final, axis=-1, direction='DESCENDING').numpy()[0:beam_width]
                for prediction in predictions:
                    formed_candidate = (beams[k][0] + [prediction],
                                        beams[k][1] + -tf.math.log(final[prediction]),
                                        new_state)
                    if len(candidates) < beam_width:
                        candidates.append(formed_candidate)
                    else:
                        for m in range(len(candidates)):
                            if candidates[m][1] > formed_candidate[1]:
                                candidates[m] = formed_candidate
                                break
        beams = candidates
        if all(beams[k][0][-1] == end_token for k in range(beam_width)):
            break
    lowest_perplexity_beam = beams[0]
    for k in range(1, beam_width):
        if beams[k][1] < lowest_perplexity_beam[1]:
            lowest_perplexity_beam = beams[k]
    return lowest_perplexity_beam


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

    @tf.function
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

    @tf.function
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
    'java': preprocess_java,
    'javadoc': preprocess_javadoc
}


class Transformer(tf.keras.Model):
    def __init__(self, model_path, input_tokenizer_path, output_tokenizer_path):
        super(Transformer, self).__init__()

        model_path = os.path.abspath(model_path)
        with open(os.path.join(model_path, "transformer_description.json")) as transformer_desc_json:
            transformer_description = json.load(transformer_desc_json)

        self.input_tokenizer = spm.SentencePieceProcessor()
        self.input_tokenizer.Load(model_file=input_tokenizer_path)
        self.input_tokenizer.SetEncodeExtraOptions('bos:eos')

        self.output_tokenizer = spm.SentencePieceProcessor()
        self.output_tokenizer.Load(model_file=output_tokenizer_path)
        self.output_tokenizer.SetEncodeExtraOptions('bos:eos')

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
                               self.input_tokenizer.vocab_size(), self.max_input_len, rate=dropout_rate,
                               universal=universal, shared_qk=shared_qk)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               self.output_tokenizer.vocab_size(), self.max_output_len, rate=dropout_rate,
                               universal=universal, shared_qk=shared_qk)

        self.final_layer = tf.keras.layers.Dense(self.output_tokenizer.vocab_size())

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

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

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

    @tf.function
    def val_loss(self, dataset):
        loss = 0.0
        i = 0
        for inp, tar in dataset:
            tar_inp = tar[:, :-1]
            tar_out = tar[:, 1:]
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
            predictions, _ = self.call(inp, tar_inp,
                                       False,
                                       enc_padding_mask,
                                       combined_mask,
                                       dec_padding_mask)
            loss += loss_function(tar_out, predictions)
            i += 1
            if i % 100 == 0:
                current_loss = loss / tf.cast(i, tf.float32)
                tf.print("Validation Step:", i, "\tCurrent Loss: ", current_loss, output_stream=sys.stdout)
        loss /= tf.cast(i, tf.float32)
        return loss

    @tf.function
    def train_one_epoch(self, train_set):
        i = 0
        for inp, tar in train_set:
            tar_inp = tar[:, :-1]
            tar_out = tar[:, 1:]
            self.train_step(inp, tar_inp, tar_out)
            i += 1
            if i % 100 == 0:
                tf.print("Train Step:", i,
                         "\tLoss:", self.train_loss.result(),
                         "\tAccuracy:", self.train_accuracy.result(),
                         output_stream=sys.stdout)

    def py_tokenize(self, code, nl):
        tok_code = self.input_tokenizer.SampleEncodeAsIds(code.numpy(), -1, 0.2)
        tok_nl = self.output_tokenizer.SampleEncodeAsIds(nl.numpy(), -1, 0.2)
        return tok_code, tok_nl

    def parallel_tokenize(self, code, nl):
        return tf.py_function(self.py_tokenize, [code, nl], [tf.int32, tf.int32])

    def filter_max_len(self, code, nl):
        return tf.logical_and(tf.size(code) <= self.max_input_len,
                              tf.size(nl) <= self.max_output_len)

    def train(self, train_codes_path, train_nl_path, val_codes_path, val_nl_path, batch_size=64, num_epochs=100, shuffle_buffer_size=10000):

        train_codes = tf.data.TextLineDataset(train_codes_path)
        train_nl = tf.data.TextLineDataset(train_nl_path)
        train_set = tf.data.Dataset.zip((train_codes, train_nl))

        val_codes = tf.data.TextLineDataset(val_codes_path)
        val_nl = tf.data.TextLineDataset(val_nl_path)
        val_set = tf.data.Dataset.zip((val_codes, val_nl))

        train_set = train_set.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        val_set = val_set.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)

        train_set = train_set.map(self.parallel_tokenize)
        val_set = val_set.map(self.parallel_tokenize)

        train_set = train_set.filter(self.filter_max_len)
        val_set = val_set.filter(self.filter_max_len)

        train_set = train_set.padded_batch(batch_size, (self.max_input_len, self.max_output_len))
        val_set = val_set.padded_batch(batch_size, (self.max_input_len, self.max_output_len))

        best_val_loss = self.val_loss(val_set)
        print('Initial Validation loss: {:.4f}'.format(best_val_loss))
        num_epochs_with_no_improvement = 0

        for epoch in range(num_epochs):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            self.train_one_epoch(train_set)

            val_loss = self.val_loss(val_set)
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

        encoder_input = self.input_tokenizer.EncodeAsIds(inp_sentence)
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

        best_beam = beam_search_decode(dec_state, self._single_bsd_step, self.output_tokenizer.bos_id(),
                                       self.output_tokenizer.eos_id(), beam_width=10, max_len=max_length)

        return best_beam[0], best_beam[2]["attention_weights"]

    def plot_attention_weights(self, attention, sentence, result):

        sentence = self.input_tokenizer.EncodeAsIds(sentence)
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
            [self.input_tokenizer.DecodeIds([i]) for i in sentence],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([self.output_tokenizer.DecodeIds([i]) for i in result_no_sos],
                           fontdict=fontdict)

        ax.set_xlabel('Encoder-Decoder Attention')

        plt.show()

    def translate(self, sentence, plot=False, print_output=False):

        sentence = self.inp_prep(sentence)
        result, attention_weights = self.evaluate_on_sentence(sentence, self.max_output_len)

        predicted_sentence = self.output_tokenizer.DecodeIds(result)

        if print_output:
            print('Input: {}'.format(sentence))
            print('Predicted translation: {}'.format(predicted_sentence))

        if plot:
            self.plot_attention_weights(attention_weights, sentence, result)

        return predicted_sentence

    def translate_batch(self, sentences, preprocessed=False):
        num_examples = len(sentences)
        if not preprocessed:
            sentences = map(self.inp_prep, sentences)
        encoder_inputs = list(map(self.input_tokenizer.EncodeAsIds, sentences))
        encoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(encoder_inputs, maxlen=self.max_input_len,
                                                                       dtype='int32', padding='post', value=0,
                                                                       truncating='post')

        enc_padding_mask = create_padding_mask(encoder_inputs)
        enc_outputs = self.encoder(encoder_inputs, False, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        tar = np.zeros((num_examples, self.max_output_len), dtype='int32')
        tar[:, 0] = self.output_tokenizer.bos_id()

        for step in range(1, self.max_output_len):
            _, combined_mask, dec_padding_mask = create_masks(encoder_inputs, tar)
            # dec_output.shape == (batch_size, tar_seq_len, d_model)
            dec_output, attention_weights = self.decoder(tar, enc_outputs, False, combined_mask,
                                                         dec_padding_mask)
            final_output = self.final_layer(dec_output)

            new_preds = tf.argmax(final_output[:, step - 1, :], axis=-1, output_type=tf.int32).numpy()
            tar[:, step] = new_preds

        de_tokenized = map(self.output_tokenizer.DecodeIds, tar)
        return de_tokenized

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
