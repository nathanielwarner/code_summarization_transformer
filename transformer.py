import tensorflow as tf
import tensorflow_text as tft

import time
import numpy as np
import json
import os
import argparse


infinity = tf.constant(np.inf)


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

    return tf.cast(pos_encoding, tf.float32)


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
        # attention_weights = []

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

            # attention_weights.append((block1, block2))

        # x.shape == (batch_size, target_seq_len, d_model)
        return x #, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, model_path, input_tokenizer_path, output_tokenizer_path, training=False):
        super(Transformer, self).__init__()

        model_path = os.path.abspath(model_path)
        with open(os.path.join(model_path, "transformer_description.json")) as transformer_desc_json:
            transformer_description = json.load(transformer_desc_json)

        if training:
            nbest_size = -1
            alpha = 0.2
        else:
            nbest_size = 0
            alpha = 1.0
        self.training = training

        in_spm = open(input_tokenizer_path, 'rb').read()
        self.input_tokenizer = tft.SentencepieceTokenizer(model=in_spm, out_type=tf.int32, nbest_size=nbest_size, alpha=alpha,
                                                          reverse=False, add_bos=True, add_eos=True)

        out_spm = open(output_tokenizer_path, 'rb').read()
        self.output_tokenizer = tft.SentencepieceTokenizer(model=out_spm, out_type=tf.int32, nbest_size=nbest_size, alpha=alpha,
                                                           reverse=False, add_bos=True, add_eos=True)
        
        in_tok_empty = self.input_tokenizer.tokenize('')
        self.in_tok_bos = in_tok_empty[0]
        self.in_tok_eos = in_tok_empty[1]

        out_tok_empty = self.output_tokenizer.tokenize('')
        self.out_tok_bos = out_tok_empty[0]
        self.out_tok_eos = out_tok_empty[1]

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
            self.restored = True
        else:
            self.restored = False

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output

    def train_step(self, inp, tar_inp, tar_out):

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions = self.call(inp, tar_inp,
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
            predictions = self.call(inp, tar_inp,
                                    False,
                                    enc_padding_mask,
                                    combined_mask,
                                    dec_padding_mask)
            loss += loss_function(tar_out, predictions)
            i += 1
            if i % 100 == 0:
                current_loss = loss / tf.cast(i, tf.float32)
                tf.print("Validation Step:", i, "\tCurrent Loss: ", current_loss)
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
                         "\tAccuracy:", self.train_accuracy.result())

    def parallel_tokenize(self, code, nl):
        return self.input_tokenizer.tokenize(code), self.output_tokenizer.tokenize(nl)
    
    def truncate_oversize_codes(self, code, nl):
        return code[:self.max_input_len], nl

    def filter_max_len(self, code, nl):
        return tf.logical_and(tf.size(code) <= self.max_input_len,
                              tf.size(nl) <= self.max_output_len)

    def train(self, train_codes_path, train_nl_path, val_codes_path, val_nl_path, batch_size=64, num_epochs=100,
              shuffle_buffer_size=10000):

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

        train_set = train_set.map(self.truncate_oversize_codes)
        val_set = val_set.map(self.truncate_oversize_codes)

        train_set = train_set.filter(self.filter_max_len)
        val_set = val_set.filter(self.filter_max_len)

        train_set = train_set.padded_batch(batch_size, (self.max_input_len, self.max_output_len))
        val_set = val_set.padded_batch(batch_size, (self.max_input_len, self.max_output_len))

        if self.restored:
            best_val_loss = self.val_loss(val_set)
        else:
            best_val_loss = tf.convert_to_tensor(float('inf'))

        print('Initial Validation loss: {:.4f}'.format(best_val_loss))
        num_epochs_with_no_improvement = 0

        for epoch in range(num_epochs):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            print("Starting epoch %d of %d" % (epoch + 1, num_epochs))
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

    def beam_search_decode(self, size_of_batch, single_bsd_step, beam_width=10):
        """
        Graph-compatible beam search decoder

        :param size_of_batch
        :param single_bsd_step: a function that takes in the current set of predictions, of shape (size_of_batch * beam_width, self.max_output_len), and returns predictions for next tokens, of shape (size_of_batch * beam_width, self.max_output_len, vocab_size).
        :param beam_width: the number of beams to keep at each step
        :return:
        """

        beam_preds = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=True, infer_shape=False)
        beam_preds = beam_preds.write(0, tf.repeat(tf.expand_dims(tf.repeat(
            tf.expand_dims(tf.expand_dims(self.out_tok_bos, 0), 0), beam_width, axis=0
        ), 0), size_of_batch, axis=0))

        beam_perps = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True, infer_shape=True)
        beam_perps = beam_perps.write(0, tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(0.0, 0), beam_width, axis=0), 0),
                                                size_of_batch, axis=0))
        
        vocab_size = self.output_tokenizer.vocab_size()

        finished_beams_excl = tf.fill((size_of_batch, beam_width, vocab_size - 1), False)

        first_iteration = True
        for step in tf.range(0, limit=self.max_output_len, delta=1):

            cur_beam_preds = beam_preds.read(0)
            cur_beam_perps = beam_perps.read(0)

            end_tokens = tf.equal(cur_beam_preds, self.out_tok_eos)
            already_finished_beams = tf.reduce_any(end_tokens, axis=-1)
            if tf.reduce_all(already_finished_beams):
                beam_preds = beam_preds.write(0, cur_beam_preds)
                break

            flat_beam_preds = tf.reshape(cur_beam_preds, (size_of_batch * beam_width, -1))
            flat_beam_preds_padded = tf.pad(flat_beam_preds, tf.convert_to_tensor(((0, 0), (0, self.max_output_len - step - 1))), mode="CONSTANT", constant_values=0)
            flat_pred_probs_pre = single_bsd_step(flat_beam_preds_padded)
            flat_pred_probs = flat_pred_probs_pre[:, step, :]
            pred_probs = tf.reshape(flat_pred_probs, (size_of_batch, beam_width, -1))

            expanded_old_perps = tf.repeat(tf.expand_dims(cur_beam_perps, -1), vocab_size, axis=-1)

            finished_beams_broadcast_with_excl = tf.concat((tf.expand_dims(already_finished_beams, 2), finished_beams_excl), 2)
            pred_perps_pre = tf.where(finished_beams_broadcast_with_excl, x=expanded_old_perps,
                                      y=expanded_old_perps - tf.math.log(pred_probs))

            finished_beams_broadcast_wo_excl = tf.repeat(tf.expand_dims(already_finished_beams, 2), vocab_size, axis=2)
            disqualified = tf.not_equal(finished_beams_broadcast_with_excl, finished_beams_broadcast_wo_excl)
            pred_perps = tf.where(disqualified, x=infinity, y=pred_perps_pre)
            
            if first_iteration:
                pred_perps = tf.expand_dims(pred_perps[:, 0, :], 1)

            new_beam_width = tf.shape(pred_perps)[1]

            flattened_pred_perps = tf.reshape(pred_perps, (size_of_batch, new_beam_width * vocab_size))
            values, indices = tf.math.top_k(-flattened_pred_perps, k=beam_width, sorted=True)
            
            true_perps = -values
            beam_perps = beam_perps.write(0, true_perps)
            first_iteration = False

            beam_indices = tf.math.floordiv(indices, vocab_size)
            chosen_beam_preds = tf.gather(cur_beam_preds, beam_indices, axis=1, batch_dims=1)
            token_indices = tf.expand_dims(indices - beam_indices * vocab_size, 2)
            new_full_beam_preds = tf.concat((chosen_beam_preds, token_indices), -1)
            beam_preds = beam_preds.write(0, new_full_beam_preds)

        best_beam_preds = beam_preds.read(0)[:, 0, :]
        return best_beam_preds

    @tf.function
    def translate_batch(self, sentences, beam_width=10):
        num_examples = sentences.shape[0]

        encoder_inputs = self.input_tokenizer.tokenize(sentences)
        encoder_inputs = encoder_inputs.to_tensor(default_value=0, shape=(num_examples, self.max_input_len))

        enc_padding_mask = create_padding_mask(encoder_inputs)
        enc_outputs = self.encoder(encoder_inputs, False, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        enc_inp_rep = tf.reshape(tf.repeat(tf.expand_dims(encoder_inputs, 1), beam_width, axis=1), (num_examples * beam_width, -1))
        enc_out_rep = tf.reshape(tf.repeat(tf.expand_dims(enc_outputs, 1), beam_width, axis=1), (num_examples * beam_width, -1, self.decoder.d_model))

        def single_bsd_step(preds):
            _, combined_mask, dec_padding_mask = create_masks(enc_inp_rep, preds)
            dec_output = self.decoder(preds, enc_out_rep, False, combined_mask, dec_padding_mask)
            final = tf.nn.softmax(self.final_layer(dec_output), axis=-1)
            return final
        
        tar = self.beam_search_decode(num_examples, single_bsd_step, beam_width=beam_width)
        tar.set_shape((num_examples, None))

        ind = tf.argmax(tf.cast(tf.equal(tar, self.out_tok_eos), tf.float32), axis=1) + 1
        ind.set_shape((num_examples,))
        tar_rag = tf.RaggedTensor.from_tensor(tar, lengths=ind)
        de_tokenized = self.output_tokenizer.detokenize(tar_rag)
        return de_tokenized

    def interactive_demo(self):
        while True:
            print()
            inp = input(">> ")
            if inp == "exit":
                break
            out = self.translate_batch(tf.convert_to_tensor([inp], dtype=tf.string))
            out = out[0].numpy().decode('utf-8')
            print("Predicted sentence: %s" % out)


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

    mask = tf.cast(mask, loss_.dtype)
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
    parser.add_argument("--dataset_path", help="Path to dataset, containing the SentencePiece"
                                           "models.", required=True, type=str)

    args = vars(parser.parse_args())
    model_path = args["model_path"]
    dataset_path = os.path.abspath(args["dataset_path"])
    code_spm_path = os.path.join(dataset_path, "code_spm.model")
    nl_spm_path = os.path.join(dataset_path, "nl_spm.model")

    transformer = Transformer(model_path, code_spm_path, nl_spm_path)
    transformer.interactive_demo()


if __name__ == "__main__":
    main()
