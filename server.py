import falcon
import os
from translation_transformer import TranslationTransformer


class Summarization:
    def __init__(self, model):
        self.model = model

    def on_post(self, req, resp):
        if req.media is not None and 'in_code' in req.media:
            in_code = req.media['in_code']
            print("Received Input Code: %s" % in_code)
            summ = self.model([in_code])[0]
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

    transformer = TranslationTransformer(model_path, code_spm_path, nl_spm_path)

    api = falcon.API()
    api.add_route('/summarize', Summarization(transformer))
    return api
