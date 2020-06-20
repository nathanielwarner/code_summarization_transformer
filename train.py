import argparse
import os
import text_data_utils as tdu
from transformer import Transformer


parser = argparse.ArgumentParser(description="Train a Transformer for code summarization")
parser.add_argument("--num_epochs", help="Number of epochs to train the model", required=True, type=int)
parser.add_argument("--model_path", help="Path to BVAE or Transformer model", required=True, type=str)
parser.add_argument("--dataset_path", help="Path to dataset", required=True, type=str)
args = vars(parser.parse_args())
model_path = args["model_path"]
num_epochs = args["num_epochs"]
dataset_path = args["dataset_path"]

print("Loading dataset...")
all_train = tdu.load_json_dataset(os.path.join(dataset_path, "train.json"))
all_val = tdu.load_json_dataset(os.path.join(dataset_path, "val.json"))

print("Creating model...")
model = Transformer(model_path, train_set=all_train, val_set=all_val, num_train_epochs=num_epochs,
                    sets_preprocessed=True)
