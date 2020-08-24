import tensorflow as tf
import sentencepiece as spm
import json


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


translator = ProdTranslator('models/java_summ_ut_4.tflite',
                            'data/leclair_java/code_spm.model', 'data/leclair_java/nl_spm.model')


def code_summarization_server(environ, start_response):
    if "CONTENT_TYPE" not in environ or environ["CONTENT_TYPE"] != "application/json":
        error = b'{"error", "Expected JSON"}'
        start_response("400 Bad Request", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(error)))
        ])
        return iter([error])
    
    if "CONTENT_LENGTH" not in environ:
        error = b'{"error", "Content length was not defined"}'
        start_response("400 Bad Request", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(error)))
        ])
        return iter([error])
    
    request_body_size = int(environ["CONTENT_LENGTH"])
    if request_body_size < 3:
        error = b'{"error", "Underfull request body"}'
        start_response("400 Bad Request", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(error)))
        ])
        return iter([error])
    
    request_body = environ["wsgi.input"].read(request_body_size).decode()
    try:
        request_json = json.loads(request_body)
    except:
        error = b'{"error", "Invalid JSON provided"}'
        start_response("400 Bad Request", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(error)))
        ])
        return iter([error])

    if not request_json or "in_code" not in request_json:
        error = b'{"error", "in_code was not defined in request body"}'
        start_response("400 Bad Request", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(error)))
        ])
        return iter([error])
    
    in_code = request_json['in_code']
    print("Input Code: %s" % in_code)
    out = translator(in_code)
    print("Generated Summary: %s\n" % out)
    response = json.dumps({"summary": out}).encode()
    start_response("200 OK", [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(response)))
    ])
    return iter([response])


def main():
    while True:
        print()
        inp = input(">> ")
        if inp == "exit" or inp == "quit":
            break
        out = translator(inp)
        print("Predicted sentence: %s" % out)


if __name__ == '__main__':
    main()
