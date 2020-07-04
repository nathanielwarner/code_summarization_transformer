import nltk
import numpy as np
import argparse
import tqdm
import os
from translation_transformer import TranslationTransformer


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

transformer = TranslationTransformer(model_path, code_spm_path, nl_spm_path)

codes_set = open(eval_codes_file, mode='r', encoding='utf-8').read().splitlines()
nl_set = open(eval_nl_file, mode='r', encoding='utf-8').read().splitlines()
assert len(codes_set) == len(nl_set)
len_set = len(codes_set)

predicts = []
meteors = []
for i in tqdm.trange(len_set):
    if print_out or i % 10 == 1:
        print("Batch %d" % (i + 1))
    if i * batch_size + batch_size < len_set:
        size_of_batch = batch_size
    else:
        size_of_batch = len_set - i
    codes_batch = codes_set[i * batch_size: i * batch_size + size_of_batch]
    nl_batch = nl_set[i * batch_size: i * batch_size + size_of_batch]
    predicts_batch = transformer(codes_batch, beam_width=beam_width)
    for j in range(size_of_batch):
        code = codes_batch[j]
        nl = nl_batch[j]
        predict = predicts_batch[j]
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

corpus_bleu_4 = nltk.translate.bleu_score.corpus_bleu(nl_set, predicts, weights=(0.25, 0.25, 0.25, 0.25))
corpus_bleu_2 = nltk.translate.bleu_score.corpus_bleu(nl_set, predicts, weights=(0.5, 0.5, 0.0, 0.0))
print("Corpus BLEU-4 score: %.4f" % corpus_bleu_4)
print("Corpus BLEU-2 score: %.4f" % corpus_bleu_2)
