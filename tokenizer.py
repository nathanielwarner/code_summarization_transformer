import os
import text_data_utils as tdu
import tensorflow_datasets as tfds
import json
from typing import List


class Tokenizer(object):

    def __init__(self, seq_type, path, target_vocab_size=None, training_texts=None):

        if seq_type == 'subwords':
            if os.path.isfile(path + ".subwords"):
                subword_encoder = tfds.features.text.SubwordTextEncoder.load_from_file(path)
            else:
                print("Could not find the tokenizer save file '%s'. Creating the tokenizer..." % (path + ".subwords"))
                subword_encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                    (tdu.eof_text(text) for text in training_texts), target_vocab_size, reserved_tokens=['<s>', '</s>']
                )
                subword_encoder.save_to_file(path)

            self.vocab_size = subword_encoder.vocab_size
            self.tokenize_text = lambda text: subword_encoder.encode(tdu.eof_text(text))
            self.de_tokenize_text = lambda seq: subword_encoder.decode(seq)
            self.start_token = subword_encoder.encode("<s>")[0]
            self.end_token = subword_encoder.encode("</s>")[0]

        else:
            raise Exception("Invalid tokenizer type %s" % seq_type)
