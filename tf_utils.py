import tensorflow as tf


def top_k_preds(pred_perps, first_iteration):

    shape = tf.shape(pred_perps)
    batch_size = shape[0]
    beam_width = shape[1]
    vocab_size = shape[2]

    if first_iteration:
        pred_perps = tf.expand_dims(pred_perps[:, 0, :], 1)

    new_beam_width = tf.shape(pred_perps)[1]

    flattened_pred_perps = tf.reshape(pred_perps, (batch_size, new_beam_width * vocab_size))
    values, indices = tf.math.top_k(-flattened_pred_perps, k=beam_width, sorted=True)
    
    true_perps = -values

    beam_indices = tf.math.floordiv(indices, vocab_size)
    token_indices = indices - beam_indices * vocab_size
    full_indices = tf.concat((tf.expand_dims(beam_indices, -1), tf.expand_dims(token_indices, -1)), -1)

    return full_indices, true_perps


#@tf.function
def beam_search_decode(size_of_batch, single_bsd_step, start_token, end_token, beam_width=10, max_len=50):
    """
    Beam search decoder

    :param size_of_batch
    :param single_bsd_step: a function that takes in the current set of predictions, of shape (size_of_batch * beam_width, step), and returns predictions for the next token, of shape (size_of_batch * beam_width, vocab_size).
    :param start_token: the tokenizer's start token
    :param end_token: the tokenizer's end token
    :param beam_width: the number of beams to keep at each step
    :param max_len: the maximum length of the decoded sequence
    :return:
    """

    beam_preds = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=True, infer_shape=False)
    beam_preds = beam_preds.write(0, tf.repeat(tf.expand_dims(tf.repeat(
        tf.expand_dims(tf.expand_dims(start_token, 0), 0), beam_width, axis=0
    ), 0), size_of_batch, axis=0))

    beam_perps = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True, infer_shape=True)
    beam_perps = beam_perps.write(0, tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(0.0, 0), beam_width, axis=0), 0),
                                               size_of_batch, axis=0))

    first_iteration = True
    for step in tf.range(0, limit=max_len, delta=1):

        cur_beam_preds = beam_preds.read(0)
        cur_beam_perps = beam_perps.read(0)

        all_end_tokens = tf.equal(cur_beam_preds, end_token)
        already_finished_beams = tf.reduce_any(all_end_tokens, axis=-1)
        if tf.reduce_all(already_finished_beams):
            break

        flat_beam_preds = tf.reshape(cur_beam_preds, (size_of_batch * beam_width, -1))
        flat_pred_probs = single_bsd_step(flat_beam_preds)
        pred_probs = tf.reshape(flat_pred_probs, (size_of_batch, beam_width, -1))

        vocab_size = tf.shape(pred_probs)[-1]
        expanded_old_perps = tf.repeat(tf.expand_dims(cur_beam_perps, -1), vocab_size, axis=-1)
        finished_beams_broadcast = tf.repeat(tf.expand_dims(already_finished_beams, 2), vocab_size, axis=2)
        conditional_pred_perps = tf.where(finished_beams_broadcast, x=expanded_old_perps,
                                          y=expanded_old_perps - tf.math.log(pred_probs))
        top_k_pred_indices, beam_perps_new = top_k_preds(conditional_pred_perps, first_iteration)
        beam_perps = beam_perps.write(0, beam_perps_new)
        first_iteration = False

        new_token_nums = tf.expand_dims(top_k_pred_indices[:, :, 1], -1)
        new_beam_preds = tf.concat((cur_beam_preds, new_token_nums), -1)
        beam_preds = beam_preds.write(0, new_beam_preds)

    best_beam_preds = beam_preds.read(0)[:, 0, :]
    return best_beam_preds
