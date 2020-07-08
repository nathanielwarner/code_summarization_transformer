# Code Summarization Transformer
Automatic summarization of source code using a neural network,
based on the Universal Transformer architecture.

Check out the [live demo](https://nathanielwarner.us/projects/code-completion-demo)!

## Details
The Transformer architecture for sequence-to-sequence modeling is comprised of an Encoder and a Decoder.
The Encoder and Decoder have sets of layers, each of which has a self-attention block and a feed-forward block. 
The Decoder layers additionally have an encoder-decoder attention block, which attends to the processed input as well as the currently generated output.

The Universal Transformer architecture uses the same encoder layer across the entire Encoder; likewise
with the Decoder. This reduces the size of the model, and improves accuracy across many tasks, including
those of an algorithmic nature (e.g. interpreting source code).

I implemented this project using TensorFlow.

## Training Locally
1. [Download the dataset and create the SentencePiece tokenizers](data/leclair_java/README.md)
2. Run the training script `train.py`, providing the parameters `num_epochs`, `model_path` (path to the model, which contains a `transformer_description.json` file with necessary attributes), and `dataset_path` (ordinarily `data/leclair_java`)

## Running Locally
If you trained a model yourself, you can run an interactive demo by running `translation_transformer.py` with the arguments `model_path` and `dataset_path`.

There is a pretrained model in the TFLite format at `models/java_summ_ut_4.tflite`, which you can try by running `main.py`. You currently need TensorFlow Nightly (`tf-nightly`), because support for all the necessary operations has not been released in the stable version. Currently only Linux is supported.

## Issues
- Need to improve documentation of the `transformer_description.json` file. Should probably switch to a standard solution like `hparams`.
