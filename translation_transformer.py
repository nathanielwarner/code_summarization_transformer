import os
import json
import sentencepiece as spm
import argparse
import tensorflow as tf
from transformer import Transformer


class TranslationTransformer(object):
    def __init__(self, model_path, input_tokenizer_path, output_tokenizer_path):

        self.inp_tokenizer = spm.SentencePieceProcessor()
        self.inp_tokenizer.LoadFromFile(input_tokenizer_path)
        self.inp_tokenizer.SetEncodeExtraOptions('bos:eos')

        self.tar_tokenizer = spm.SentencePieceProcessor()
        self.tar_tokenizer.LoadFromFile(output_tokenizer_path)
        self.tar_tokenizer.SetEncodeExtraOptions('bos:eos')

        model_path = os.path.abspath(model_path)
        with open(os.path.join(model_path, 'transformer_description.json')) as transformer_desc_json:
            desc = json.load(transformer_desc_json)
        checkpoint_path = os.path.join(model_path, 'train')

        self.inp_dim = desc['inp_dim']
        self.tar_dim = desc['tar_dim']

        self.model = Transformer(num_layers=desc['num_layers'], d_model=desc['d_model'], dff=desc['dff'],
                                 num_heads=desc['num_heads'], dropout_rate=desc['dropout_rate'],
                                 universal=desc['universal'], shared_qk=desc['shared_qk'],
                                 inp_dim=self.inp_dim, inp_vocab_size=self.inp_tokenizer.vocab_size(),
                                 tar_dim=self.tar_dim, tar_vocab_size=self.tar_tokenizer.vocab_size(),
                                 inp_bos=self.inp_tokenizer.bos_id(), inp_eos=self.inp_tokenizer.eos_id(),
                                 tar_bos=self.tar_tokenizer.bos_id(), tar_eos=self.tar_tokenizer.eos_id(),
                                 ckpt_path=checkpoint_path)

    def parallel_tokenize_py(self, inp, tar):
        return self.inp_tokenizer.SampleEncodeAsIds(inp.numpy(), -1, 0.2),\
               self.tar_tokenizer.SampleEncodeAsIds(tar.numpy(), -1, 0.2)

    def parallel_tokenize(self, inp, tar):
        return tf.py_function(self.parallel_tokenize_py, [inp, tar], [tf.int32, tf.int32])

    def truncate_oversize_inputs(self, inp, tar):
        return inp[:self.inp_dim], tar

    def filter_max_len(self, inp, tar):
        return tf.logical_and(tf.size(inp) <= self.inp_dim,
                              tf.size(tar) <= self.tar_dim)

    def create_parallel_dataset(self, inp_path, tar_path, batch_size=64, shuffle_buffer_size=10000,
                                filter_oversize_targets=False):
        inp = tf.data.TextLineDataset(inp_path)
        tar = tf.data.TextLineDataset(tar_path)
        dataset = tf.data.Dataset.zip((inp, tar))
        dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.map(self.parallel_tokenize)
        dataset = dataset.map(self.truncate_oversize_inputs)
        if filter_oversize_targets:
            dataset = dataset.filter(self.filter_max_len)
        dataset = dataset.padded_batch(batch_size, (self.inp_dim, self.tar_dim))
        return dataset

    def train(self, train_inp_path, train_tar_path, val_inp_path, val_tar_path, batch_size=64, num_epochs=100,
              shuffle_buffer_size=10000):

        train_set = self.create_parallel_dataset(train_inp_path, train_tar_path, batch_size=batch_size,
                                                 shuffle_buffer_size=shuffle_buffer_size, filter_oversize_targets=True)
        val_set = self.create_parallel_dataset(val_inp_path, val_tar_path, batch_size=batch_size,
                                               shuffle_buffer_size=shuffle_buffer_size, filter_oversize_targets=True)

        self.model.train(train_set, val_set, num_epochs=num_epochs)

    def __call__(self, inputs, beam_width=10):
        inp_tok = [self.inp_tokenizer.EncodeAsIds(inp) for inp in inputs]
        inp_pad = tf.keras.preprocessing.sequence.pad_sequences(inp_tok, maxlen=self.inp_dim, padding='post',
                                                                truncating='post', dtype='int32')
        inp_tensor = tf.convert_to_tensor(inp_pad)
        pred_tar = self.model.translate_batch(inp_tensor, beam_width=beam_width).numpy()
        ends = tf.argmax(tf.cast(tf.equal(pred_tar, self.tar_tokenizer.eos_id()), tf.float32), axis=1).numpy() + 1
        pred_detok = []
        for i in range(len(pred_tar)):
            pred_detok.append(self.tar_tokenizer.DecodeIds(pred_tar[i, :ends[i]].tolist()))
        return pred_detok


def main():
    parser = argparse.ArgumentParser(description="Demonstrate the code summarization abilities of the Transformer")
    parser.add_argument("--model_path", help="Path to the Transformer model", required=True)
    parser.add_argument("--dataset_path", help="Path to dataset, containing the SentencePiece"
                                               "models.", required=True, type=str)

    args = vars(parser.parse_args())
    model_path = args["model_path"]
    dataset_path = os.path.abspath(args["dataset_path"])
    code_spm_path = os.path.join(dataset_path, "code_spm.model")
    nl_spm_path = os.path.join(dataset_path, "nl_spm.model")

    model = TranslationTransformer(model_path, code_spm_path, nl_spm_path)

    while True:
        print()
        inp = input(">> ")
        if inp == "exit" or inp == "quit":
            break
        out = model([inp])[0]
        print("Predicted sentence: %s" % out)


if __name__ == '__main__':
    main()
