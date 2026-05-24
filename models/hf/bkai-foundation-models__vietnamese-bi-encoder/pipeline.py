from typing import Dict, List, Union
import torch
from transformers import AutoModel
from custom_tokenizer import CustomPhobertTokenizer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class PreTrainedPipeline:
    def __init__(self, path="."):
        self.model = AutoModel.from_pretrained(path)
        self.tokenizer = CustomPhobertTokenizer.from_pretrained(path)

    def __call__(self, inputs: Dict[str, Union[str, List[str]]]) -> List[float]:
        """
        Args:
            inputs (Dict[str, Union[str, List[str]]]):
                a dictionary containing a query sentence and a list of key sentences
        """

        # Combine the query sentence and key sentences into one list
        sentences = [inputs["source_sentence"]] + inputs["sentences"]

        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling to get sentence embeddings
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Separate the query embedding from the key embeddings
        query_embedding = sentence_embeddings[0]
        key_embeddings = sentence_embeddings[1:]

        # Compute cosine similarities (or any other comparison method you prefer)
        cosine_similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), key_embeddings
        )

        # Convert the tensor of cosine similarities to a list of floats
        scores = cosine_similarities.tolist()

        return scores


if __name__ == "__main__":
    inputs = {
        "source_sentence": "Anh ấy đang là sinh viên năm cuối",
        "sentences": [
            "Anh ấy học tại Đại học Bách khoa Hà Nội, chuyên ngành Khoa học máy tính",
            "Anh ấy đang làm việc tại nhà máy sản xuất linh kiện điện tử",
            "Anh ấy chuẩn bị đi du học nước ngoài",
            "Anh ấy sắp mở cửa hàng bán mỹ phẩm",
            "Nhà anh ấy có rất nhiều cây cảnh",
        ],
    }

    pipeline = PreTrainedPipeline()
    res = pipeline(inputs)
