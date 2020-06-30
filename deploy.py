import tensorflow as tf
import argparse
import os

from transformer import Transformer


parser = argparse.ArgumentParser(description="Deploy a Transformer to the SavedModel format")
parser.add_argument("--model_path", help="Path to the Transformer model", required=True)
parser.add_argument("--dataset_path", help="Path to dataset, containing the SentencePiece"
                                           "models.", required=True, type=str)
parser.add_argument("--output_path", help="Path to write the SavedModel", required=True, type=str)

args = vars(parser.parse_args())
model_path = args["model_path"]
dataset_path = os.path.abspath(args["dataset_path"])
output_path = os.path.abspath(args["output_path"])

code_spm_path = os.path.join(dataset_path, "code_spm.model")
nl_spm_path = os.path.join(dataset_path, "nl_spm.model")

transformer = Transformer(model_path, code_spm_path, nl_spm_path)

tf.saved_model.save(transformer, output_path)
