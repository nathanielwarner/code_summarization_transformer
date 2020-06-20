import nltk
import numpy as np
import text_data_utils as tdu
import argparse
import tqdm
from transformer import Transformer


nltk.download('wordnet')

parser = argparse.ArgumentParser(description="Evaluate a model's source code summarization")

parser.add_argument("--model_path", help="Path to the model", required=True)
parser.add_argument("--eval_set_path", help="Path to the set to evaluate on", required=True)
parser.add_argument("--batch_size", help="Size of batches to feed into the model", type=int, default=64)
parser.add_argument("--print_out", help="Print correct and predicted translation, and METEOR score for every sentence",
                    type=bool, default=False)
args = vars(parser.parse_args())

model_path = args["model_path"]
eval_set_path = args["eval_set_path"]
batch_size = args["batch_size"]
print_out = args["print_out"]

transformer = Transformer(model_path)

dataset = tdu.load_json_dataset(eval_set_path)
len_dataset = len(dataset)
codes = [ex[1] for ex in dataset]
true_summaries = [[ex[0]] for ex in dataset]

num_batches = int((len(codes) / batch_size) + 1)
predicts = []
meteors = []
for batch_num in tqdm.trange(num_batches):
    codes_batch = codes[batch_num * batch_size: batch_num * batch_size + batch_size]
    size_of_batch = len(codes_batch)
    predicts_batch = transformer.translate_batch(codes_batch, preprocessed=True)
    predicts.extend(predicts_batch)
    for i in range(batch_num * batch_size, batch_num * batch_size + size_of_batch):
        if print_out:
            print("%d of %d" % (i, len_dataset))
            print("Code: %s" % codes[i])
            print("True Summaries: %s" % true_summaries[i])
            print("Predicted Summary: %s" % predicts[i])
        meteor = nltk.translate.meteor_score.meteor_score(true_summaries[i], predicts[i])
        if print_out:
            print("METEOR score: %.4f" % meteor)
            print()
        meteors.append(meteor)

average_meteor = np.mean(meteors)
print("Average METEOR score: %.4f\n" % average_meteor)

corpus_bleu_4 = nltk.translate.bleu_score.corpus_bleu(true_summaries, predicts, weights=(0.25, 0.25, 0.25, 0.25))
corpus_bleu_2 = nltk.translate.bleu_score.corpus_bleu(true_summaries, predicts, weights=(0.5, 0.5, 0.0, 0.0))
print("Corpus BLEU-4 score: %.4f" % corpus_bleu_4)
print("Corpus BLEU-2 score: %.4f" % corpus_bleu_2)
