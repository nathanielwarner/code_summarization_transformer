import tensorflow as tf
import nltk
import numpy as np
import argparse
import tqdm
import os
from transformer import Transformer


nltk.download('wordnet')

parser = argparse.ArgumentParser(description="Evaluate a model's source code summarization")

parser.add_argument("--model_path", help="Path to the model", required=True, type=str)
parser.add_argument("--dataset_path", help="Path to the dataset, which contains the eval set as well the tokenizers", required=True, type=str)
parser.add_argument("--eval_set_filename_prefix", help="Prefix of the filename of the dataset to evaluate on", required=True, type=str)
parser.add_argument("--beam_width", help="Beam width for beam search decoding", type=int, default=10)
parser.add_argument("--batch_size", help="Size of batches to feed into the model", type=int, default=64)
parser.add_argument("--print_out", help="Print correct and predicted translation, and METEOR score for every sentence",
                    type=bool, default=False)
args = vars(parser.parse_args())

model_path = args["model_path"]
dataset_path = os.path.abspath(args["dataset_path"])
eval_set_prefix = args["eval_set_filename_prefix"]
eval_codes_file = os.path.join(dataset_path, eval_set_prefix + "_codes.txt")
eval_nl_file = os.path.join(dataset_path, eval_set_prefix + "_nl.txt")
beam_width = args["beam_width"]
batch_size = args["batch_size"]
print_out = args["print_out"]

code_spm_path = os.path.join(dataset_path, "code_spm.model")
nl_spm_path = os.path.join(dataset_path, "nl_spm.model")

transformer = Transformer(model_path, code_spm_path, nl_spm_path)

codes_set = tf.data.TextLineDataset(eval_codes_file)
nl_set = tf.data.TextLineDataset(eval_nl_file)
eval_set = tf.data.Dataset.zip((codes_set, nl_set))
eval_set = eval_set.batch(batch_size)

trues = []
predicts = []
meteors = []
batch_num = 0
for codes_batch, nl_batch in eval_set:
    batch_num += 1
    if print_out or batch_num % 10 == 0:
        print("Batch %d" % batch_num)
    size_of_batch = tf.size(codes_batch).numpy()
    predicts_batch = transformer.translate_batch(codes_batch, beam_width=beam_width)
    for i in range(size_of_batch):
        code = codes_batch[i].numpy().decode('utf-8')
        nl = nl_batch[i].numpy().decode('utf-8')
        predict = predicts_batch[i].numpy().decode('utf-8')

        trues.append([nl])
        predicts.append(predict)
        
        meteor = nltk.translate.meteor_score.meteor_score([nl], predict)
        if print_out:
            print("\nCode: %s" % code)
            print("True Summary: %s" % nl)
            print("Predicted Summary: %s" % predict)
            print("METEOR score: %.4f\n" % meteor)
        meteors.append(meteor)

average_meteor = np.mean(meteors)
print("Average METEOR score: %.4f\n" % average_meteor)

corpus_bleu_4 = nltk.translate.bleu_score.corpus_bleu(trues, predicts, weights=(0.25, 0.25, 0.25, 0.25))
corpus_bleu_2 = nltk.translate.bleu_score.corpus_bleu(trues, predicts, weights=(0.5, 0.5, 0.0, 0.0))
print("Corpus BLEU-4 score: %.4f" % corpus_bleu_4)
print("Corpus BLEU-2 score: %.4f" % corpus_bleu_2)
