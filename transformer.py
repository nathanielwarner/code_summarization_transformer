import tensorflow as tf

import time
import numpy as np

from beam_search_decode import beam_search_decode


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

        # x.shape == (batch_size, target_seq_len, d_model)
        return x


class Transformer(tf.Module):
    def __init__(self, num_layers=4, d_model=128, dff=512, num_heads=8, dropout_rate=0.1,
                 universal=True, shared_qk=True, inp_dim=80, inp_vocab_size=None, tar_dim=40, tar_vocab_size=None,
                 inp_bos=None, inp_eos=None, tar_bos=None, tar_eos=None, ckpt_path=None):
        super(Transformer, self).__init__()

        assert inp_vocab_size and tar_vocab_size and inp_bos and inp_eos and tar_bos and tar_eos and ckpt_path

        self.tar_bos = tar_bos
        self.tar_eos = tar_eos
        self.out_tok_eos = tar_eos
        self.tar_vocab_size = tar_vocab_size
        self.tar_dim = tar_dim
        self.inp_dim = inp_dim

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               inp_vocab_size, inp_dim, rate=dropout_rate,
                               universal=universal, shared_qk=shared_qk)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               tar_vocab_size, tar_dim, rate=dropout_rate,
                               universal=universal, shared_qk=shared_qk)

        self.final_layer = tf.keras.layers.Dense(tar_vocab_size)

        learning_rate = CustomSchedule(d_model)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                                  epsilon=1e-9)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        ckpt = tf.train.Checkpoint(transformer=self,
                                   optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=5)

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

    def train(self, train_set, val_set, num_epochs=100):

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

    @tf.function
    def translate_batch(self, encoder_inputs, beam_width=10):
        num_examples = encoder_inputs.get_shape()[0]
        encoder_inputs.set_shape((num_examples, self.inp_dim))

        enc_padding_mask = create_padding_mask(encoder_inputs)
        enc_outputs = self.encoder(encoder_inputs, False, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        enc_inp_rep = tf.reshape(tf.repeat(tf.expand_dims(encoder_inputs, 1), beam_width, axis=1), (num_examples * beam_width, -1))
        enc_out_rep = tf.reshape(tf.repeat(tf.expand_dims(enc_outputs, 1), beam_width, axis=1), (num_examples * beam_width, -1, self.decoder.d_model))

        def single_bsd_step(preds):
            _, combined_mask, dec_padding_mask = create_masks(enc_inp_rep, preds)
            dec_output = self.decoder(preds, enc_out_rep, False, combined_mask, dec_padding_mask)
            final = tf.nn.softmax(self.final_layer(dec_output), axis=-1)
            return final
        
        tar = beam_search_decode(num_examples, single_bsd_step, self.tar_dim, self.tar_bos, self.tar_eos,
                                 self.tar_vocab_size, beam_width=beam_width)
        tar.set_shape((num_examples, self.tar_dim))
        return tar


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
