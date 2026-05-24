from transformers import PhobertTokenizer
from pyvi import ViTokenizer


class CustomPhobertTokenizer(PhobertTokenizer):
    def rdr_segment(self, text):
        return ViTokenizer.tokenize(text)

    def _tokenize(self, text):
        segmented_text = self.rdr_segment(text)
        return super()._tokenize(segmented_text)
