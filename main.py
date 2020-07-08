import tensorflow as tf
import sentencepiece as spm
import flask


translator = None


class ProdTranslator:
    def __init__(self, model_path, inp_tok_path, tar_tok_path):
        self.inp_tokenizer = spm.SentencePieceProcessor()
        self.inp_tokenizer.LoadFromFile(inp_tok_path)
        self.inp_tokenizer.SetEncodeExtraOptions('bos:eos')

        self.tar_tokenizer = spm.SentencePieceProcessor()
        self.tar_tokenizer.LoadFromFile(tar_tok_path)
        self.tar_tokenizer.SetEncodeExtraOptions('bos:eos')

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.inp_dim = self.input_details[0]['shape'][1]

    def __call__(self, inp, **kwargs):
        inp_tok = self.inp_tokenizer.EncodeAsIds(inp)
        inp_pad = tf.keras.preprocessing.sequence.pad_sequences([inp_tok], dtype='int32', maxlen=self.inp_dim,
                                                                padding='post', truncating='post')
        self.interpreter.set_tensor(self.input_details[0]['index'], inp_pad)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        end = tf.argmax(tf.cast(tf.equal(out, self.tar_tokenizer.eos_id()), tf.float32), axis=0).numpy() + 1
        out_detok = self.tar_tokenizer.DecodeIds(out[:end].tolist())
        return out_detok


def code_summarization_server(request: flask.Request):
    global translator
    if translator is None:
        translator = ProdTranslator('models/java_summ_ut_4.tflite',
                                    'data/leclair_java/code_spm.model', 'data/leclair_java/nl_spm.model')
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'in_code' in request_json:
        in_code = request_json['in_code']
        out = translator(in_code)
        return {'summary': out}, 200, {'Content-Type': 'application/json'}
    else:
        return {'summary': 'Invalid Request'}, 500, {'Content-Type': 'application/json'}


def main():
    global translator
    translator = ProdTranslator('models/java_summ_ut_4.tflite',
                                'data/leclair_java/code_spm.model', 'data/leclair_java/nl_spm.model')
    while True:
        print()
        inp = input(">> ")
        if inp == "exit" or inp == "quit":
            break
        out = translator(inp)
        print("Predicted sentence: %s" % out)


if __name__ == '__main__':
    main()
