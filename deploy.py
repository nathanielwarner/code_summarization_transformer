import tensorflow as tf
import argparse
import os

from translation_transformer import TranslationTransformer


parser = argparse.ArgumentParser(description="Deploy a Transformer to the SavedModel format")
parser.add_argument("--model_path", help="Path to the Transformer model", required=True)
parser.add_argument("--dataset_path", help="Path to dataset, containing the SentencePiece"
                                           "models.", required=True, type=str)
parser.add_argument("--output_path", help="Path to deploy the model", required=True, type=str)

args = vars(parser.parse_args())
model_path = args["model_path"]
dataset_path = os.path.abspath(args["dataset_path"])
output_path = os.path.abspath(args["output_path"])

code_spm_path = os.path.join(dataset_path, "code_spm.model")
nl_spm_path = os.path.join(dataset_path, "nl_spm.model")

translation_transformer = TranslationTransformer(model_path, code_spm_path, nl_spm_path)

call = translation_transformer.model.translate_batch.get_concrete_function(
    tf.TensorSpec((1, translation_transformer.inp_dim), dtype=tf.int32)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([call])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

with tf.io.gfile.GFile(output_path, 'wb') as f:
    f.write(tflite_model)
