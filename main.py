import tensorflow as tf
import sentencepiece as spm
import flask


translator = None


class ProdTranslationServer:
    def __init__(self, model_path, inp_tok_path, tar_tok_path):
        self.inp_tokenizer = spm.SentencePieceProcessor()
        self.inp_tokenizer.LoadFromFile(inp_tok_path)
        self.inp_tokenizer.SetEncodeExtraOptions('bos:eos')

        self.tar_tokenizer = spm.SentencePieceProcessor()
        self.tar_tokenizer.LoadFromFile(tar_tok_path)
        self.tar_tokenizer.SetEncodeExtraOptions('bos:eos')

        self.model = tf.saved_model.load(model_path)

        self.inp_dim = self.model.translate_batch.concrete_functions[0].structured_input_signature[0][0].shape[1]

    def __call__(self, inp, **kwargs):
        inp_tok = self.inp_tokenizer.EncodeAsIds(inp)
        inp_pad = tf.keras.preprocessing.sequence.pad_sequences([inp_tok], dtype='int32', maxlen=self.inp_dim,
                                                                padding='post', truncating='post')
        out = self.model.translate_batch(inp_pad).numpy()[0]
        end = tf.argmax(tf.cast(tf.equal(out, self.tar_tokenizer.eos_id()), tf.float32), axis=0).numpy() + 1
        out_detok = self.tar_tokenizer.DecodeIds(out[:end].tolist())
        return out_detok


def code_summarization_server(request: flask.Request):
    global translator
    if translator is None:
        translator = ProdTranslationServer('models/java_summ_ut_prod1/1',
                                           'data/leclair_java/code_spm.model', 'data/leclair_java/nl_spm.model')
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'in_code' in request_json:
        in_code = request_json['in_code']
        out = translator(in_code)
        return {'summary': out}, 200, {'Content-Type': 'application/json'}
    else:
        return {'summary': 'Invalid Request'}, 500, {'Content-Type': 'application/json'}
