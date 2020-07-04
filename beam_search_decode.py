import tensorflow as tf


infinity = tf.convert_to_tensor(float('inf'))


def beam_search_decode(size_of_batch, single_bsd_step, tar_dim, tar_bos, tar_eos, tar_vocab_size, beam_width=10):
    """
    Graph-compatible beam search decoder

    :param size_of_batch
    :param single_bsd_step: a function that takes in the current set of predictions, of shape (size_of_batch * beam_width, self.max_output_len), and returns predictions for next tokens, of shape (size_of_batch * beam_width, self.max_output_len, vocab_size).
    :param tar_dim
    :param tar_bos
    :param tar_eos
    :param tar_vocab_size
    :param beam_width: the number of beams to keep at each step
    :return:
    """

    beam_preds = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=True, infer_shape=False)
    beam_preds = beam_preds.write(0, tf.repeat(tf.expand_dims(tf.repeat(
        tf.expand_dims(tf.expand_dims(tar_bos, 0), 0), beam_width, axis=0
    ), 0), size_of_batch, axis=0))

    beam_perps = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True, infer_shape=True)
    beam_perps = beam_perps.write(0, tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(0.0, 0), beam_width, axis=0), 0),
                                               size_of_batch, axis=0))

    finished_beams_excl = tf.fill((size_of_batch, beam_width, tar_vocab_size - 1), False)

    first_iteration = True
    for step in tf.range(0, limit=tar_dim, delta=1):

        cur_beam_preds = beam_preds.read(0)
        cur_beam_perps = beam_perps.read(0)

        end_tokens = tf.equal(cur_beam_preds, tar_eos)
        already_finished_beams = tf.reduce_any(end_tokens, axis=-1)
        if tf.reduce_all(already_finished_beams):
            beam_preds = beam_preds.write(0, cur_beam_preds)
            break

        flat_beam_preds = tf.reshape(cur_beam_preds, (size_of_batch * beam_width, -1))
        flat_beam_preds_padded = tf.pad(flat_beam_preds, tf.convert_to_tensor(((0, 0), (0, tar_dim - step - 1))),
                                        mode="CONSTANT", constant_values=0)
        flat_pred_probs_pre = single_bsd_step(flat_beam_preds_padded)
        flat_pred_probs = flat_pred_probs_pre[:, step, :]
        pred_probs = tf.reshape(flat_pred_probs, (size_of_batch, beam_width, -1))

        expanded_old_perps = tf.repeat(tf.expand_dims(cur_beam_perps, -1), tar_vocab_size, axis=-1)

        finished_beams_broadcast_with_excl = tf.concat((tf.expand_dims(already_finished_beams, 2), finished_beams_excl),
                                                       2)
        pred_perps_pre = tf.where(finished_beams_broadcast_with_excl, x=expanded_old_perps,
                                  y=expanded_old_perps - tf.math.log(pred_probs))

        finished_beams_broadcast_wo_excl = tf.repeat(tf.expand_dims(already_finished_beams, 2), tar_vocab_size, axis=2)
        disqualified = tf.not_equal(finished_beams_broadcast_with_excl, finished_beams_broadcast_wo_excl)
        pred_perps = tf.where(disqualified, x=infinity, y=pred_perps_pre)

        if first_iteration:
            pred_perps = tf.expand_dims(pred_perps[:, 0, :], 1)

        new_beam_width = tf.shape(pred_perps)[1]

        flattened_pred_perps = tf.reshape(pred_perps, (size_of_batch, new_beam_width * tar_vocab_size))
        values, indices = tf.math.top_k(-flattened_pred_perps, k=beam_width, sorted=True)

        true_perps = -values
        beam_perps = beam_perps.write(0, true_perps)
        first_iteration = False

        beam_indices = tf.math.floordiv(indices, tar_vocab_size)
        chosen_beam_preds = tf.gather(cur_beam_preds, beam_indices, axis=1, batch_dims=1)
        token_indices = tf.expand_dims(indices - beam_indices * tar_vocab_size, 2)
        new_full_beam_preds = tf.concat((chosen_beam_preds, token_indices), -1)
        beam_preds = beam_preds.write(0, new_full_beam_preds)

    best_beam_preds = beam_preds.read(0)[:, 0, :]
    return best_beam_preds
