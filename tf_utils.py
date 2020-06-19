import random
import tensorflow as tf


def dataset_to_batched_tensors(dataset, batch_size, tar_dim, inp_dim):
    batch_cutoffs = range(0, len(dataset), batch_size)
    num_batches = len(batch_cutoffs) - 1

    random.seed()
    random.shuffle(dataset)

    def generator():
        for i in range(num_batches):
            batch = dataset[batch_cutoffs[i]: batch_cutoffs[i + 1]]
            tar = [ex[0] for ex in batch]
            inp = [ex[1] for ex in batch]
            summaries = tf.keras.preprocessing.sequence.pad_sequences(tar, maxlen=tar_dim, padding='post',
                                                                      truncating='post', dtype='int32')
            codes = tf.keras.preprocessing.sequence.pad_sequences(inp, maxlen=inp_dim, padding='post',
                                                                  truncating='post', dtype='int32')
            summaries = tf.convert_to_tensor(summaries)
            codes = tf.convert_to_tensor(codes)
            yield summaries, codes

    return generator(), num_batches


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


@tf.function
def beam_search_decode_new(initial_states, single_bsd_step, start_token, end_token, beam_width=10, max_len=50):
    """
    Beam search decoder

    :param initial_states: shape (num_to_decode, arbitrary...)
    :param single_bsd_step: a function that takes in the current set of predictions, of shape (size_of_batch * beam_width, step), along with the current state, of shape (size_of_batch * beam_width, arbitrary...), and returns predictions for the next token, of shape (size_of_batch * beam_width, vocab_size), and the new state, of shape (size_of_batch * beam_width, arbitrary...). The state is not modified by beam_search_decode, and is passed along to the next call of single_bsd_step. It can be used for a recurrent cell's state, for example.
    :param start_token: the tokenizer's start token
    :param end_token: the tokenizer's end token
    :param beam_width: the number of beams to keep at each step
    :param max_len: the maximum length of the decoded sequence
    :return:
    """

    size_of_batch = tf.shape(initial_states)[0]

    beam_states = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True, infer_shape=True)
    beam_states = beam_states.write(0, tf.repeat(tf.expand_dims(initial_states, 1), beam_width, axis=1))

    beam_preds = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=True, infer_shape=False)
    beam_preds = beam_preds.write(0, tf.repeat(tf.expand_dims(tf.repeat(
        tf.expand_dims(tf.expand_dims(start_token, 0), 0), beam_width, axis=0
    ), 0), size_of_batch, axis=0))

    beam_perps = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True, infer_shape=True)
    beam_perps = beam_perps.write(0, tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(0.0, 0), beam_width, axis=0), 0),
                                               size_of_batch, axis=0))

    first_iteration = True
    for step in tf.range(0, limit=max_len, delta=1):

        cur_beam_states = beam_states.read(0)
        cur_beam_preds = beam_preds.read(0)
        cur_beam_perps = beam_perps.read(0)

        all_end_tokens = tf.equal(cur_beam_preds, end_token)
        already_finished_beams = tf.reduce_any(all_end_tokens, axis=-1)
        if tf.reduce_all(already_finished_beams):
            break

        flat_beam_preds = tf.reshape(cur_beam_preds, (size_of_batch * beam_width, -1))
        flat_beam_states = tf.reshape(cur_beam_states, (size_of_batch * beam_width, -1))
        flat_pred_probs, flat_new_beam_states = single_bsd_step(flat_beam_preds, flat_beam_states)
        pred_probs = tf.reshape(flat_pred_probs, (size_of_batch, beam_width, -1))
        new_beam_states = tf.reshape(flat_new_beam_states, (size_of_batch, beam_width, -1))

        vocab_size = tf.shape(pred_probs)[-1]
        expanded_old_perps = tf.repeat(tf.expand_dims(cur_beam_perps, -1), vocab_size, axis=-1)
        finished_beams_broadcast = tf.repeat(tf.expand_dims(already_finished_beams, 2), vocab_size, axis=2)
        conditional_pred_perps = tf.where(finished_beams_broadcast, x=expanded_old_perps,
                                          y=expanded_old_perps - tf.math.log(pred_probs))
        top_k_pred_indices, beam_perps_new = top_k_preds(conditional_pred_perps, first_iteration)
        beam_perps = beam_perps.write(0, beam_perps_new)
        first_iteration = False

        beam_nums = top_k_pred_indices[:, :, 0]
        beam_states = beam_states.write(0, tf.gather(new_beam_states, beam_nums, batch_dims=1))
        new_token_nums = tf.expand_dims(top_k_pred_indices[:, :, 1], -1)
        new_beam_preds = tf.concat((cur_beam_preds, new_token_nums), -1)
        beam_preds = beam_preds.write(0, new_beam_preds)

    best_beam_preds = beam_preds.read(0)[:, 0, :]
    return best_beam_preds


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
