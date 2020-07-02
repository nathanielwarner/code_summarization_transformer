import falcon
import os
import tensorflow as tf
from transformer import Transformer


class Summarization:
    def __init__(self, model):
        self.model = model

    def on_post(self, req, resp):
        if req.params is not None and 'in_code' in req.params:
            in_code = req.params['in_code']
            print("Received Input Code: %s" % in_code)
            in_code_tensor = tf.convert_to_tensor([in_code], dtype=tf.string)
            summ = self.model.translate_batch(in_code_tensor)[0].numpy().decode('utf-8')
            print("Generated Completion: %s\n" % summ)
        else:
            print("Received Invalid Request: %s\n" % req)
            summ = 'Error!!'
        resp.media = {
            'summarization': summ
        }
        


def build_server(model_path, dataset_path):
    dataset_path = os.path.abspath(dataset_path)
    code_spm_path = os.path.join(dataset_path, "code_spm.model")
    nl_spm_path = os.path.join(dataset_path, "nl_spm.model")

    transformer = Transformer(model_path, code_spm_path, nl_spm_path)

    api = falcon.API()
    api.add_route('/summarize', Summarization(transformer))
    return api
