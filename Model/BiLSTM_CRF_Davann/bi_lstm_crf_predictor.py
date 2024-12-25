from bi_lstm_crf_helper import BiLSTM_CRF
from transformers import AutoTokenizer
import torch
class BiLSTM_CRF_Predictor:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BiLSTM_CRF_Predictor, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
        self.model = BiLSTM_CRF(43050, {'O': 0, 'B': 1, 'I': 2, '<START>': 3, '<STOP>': 4}, 5, 4)
        load_model = torch.load("bi_lstm_crf_model.pt")
        self.model.load_state_dict(load_model)

    def words_to_numbers(self,words,min=30):
        words = self.tokenizer(words).input_ids
        return torch.tensor(words)

    def prediction(self,sentence=""):
        feature_words = self.words_to_numbers(sentence)
        with torch.no_grad():
            return self.model(feature_words)[1],self.tokenizer.convert_ids_to_tokens(feature_words)