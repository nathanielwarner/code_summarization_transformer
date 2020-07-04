import argparse
import os
from translation_transformer import TranslationTransformer


parser = argparse.ArgumentParser(description="Train a Transformer for code summarization")
parser.add_argument("--num_epochs", help="Number of epochs to train the model", required=True, type=int)
parser.add_argument("--model_path", help="Path to BVAE or Transformer model", required=True, type=str)
parser.add_argument("--dataset_path", help="Path to dataset, containing train_codes.txt, ..., as well as SentencePiece"
                                           "models.", required=True, type=str)
args = vars(parser.parse_args())
model_path = os.path.abspath(args["model_path"])
num_epochs = args["num_epochs"]
dataset_path = os.path.abspath(args["dataset_path"])

code_spm_path = os.path.join(dataset_path, "code_spm.model")
nl_spm_path = os.path.join(dataset_path, "nl_spm.model")

train_codes_path = os.path.join(dataset_path, "train_codes.txt")
train_nl_path = os.path.join(dataset_path, "train_nl.txt")
val_codes_path = os.path.join(dataset_path, "val_codes.txt")
val_nl_path = os.path.join(dataset_path, "val_nl.txt")

model = TranslationTransformer(model_path, code_spm_path, nl_spm_path)
model.train(train_codes_path, train_nl_path, val_codes_path, val_nl_path, num_epochs=num_epochs)
