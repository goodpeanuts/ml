import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class LogicRgeressionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, input_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(embedding_dim * input_length, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
asset_dir = os.path.join("./", "assets/")
os.makedirs(asset_dir, exist_ok=True)

tokenizer_path = os.path.join(asset_dir, "tokenizer.json")
project_dir = "../"
imdb_dataset_dir = os.path.join(project_dir, "dataset/aclImdb/")
training_path = os.path.join(imdb_dataset_dir, "train/")
test_path = os.path.join(imdb_dataset_dir, "test/")

training_paths = [
    os.path.join(training_path, "neg/"),
    os.path.join(training_path, "pos/"),
]
test_paths = [
    os.path.join(test_path, "neg/"),
    os.path.join(test_path, "pos/"),
]
class ImdbDataset(Dataset):
    def __init__(self, max_length: int, is_train: bool):
        self.max_length = max_length
        self.data = []
        target_paths = training_paths if is_train else test_paths
        tokenizer = Tokenizer.from_file(tokenizer_path)
        for label, target_path in enumerate(target_paths):
            for file_name in os.listdir(target_path):
                with open(
                    os.path.join(target_path, file_name), mode="r", encoding="utf8"
                ) as file:
                    self.data.append((tokenizer.encode(file.read().lower()).ids, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self._aligner(self.data[idx][0]),
            torch.tensor(self.data[idx][1], dtype=torch.float32),
        )

    def _aligner(self, sentence: list) -> torch.Tensor:
        sentence_tensor = torch.tensor(sentence, dtype=torch.long)
        if len(sentence_tensor) > self.max_length:
            sentence_tensor = sentence_tensor[: self.max_length]
        if len(sentence_tensor) < self.max_length:
            padding = torch.zeros(
                self.max_length - len(sentence_tensor), dtype=torch.long
            )
            sentence_tensor = torch.cat([sentence_tensor, padding])
        return sentence_tensor



def read_text_from_path(paths):
    data = []
    for folder in paths:
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data.extend([line.lower() for line in f.read().splitlines()])
    return data


def create_tokenizer():
    data = read_text_from_path(training_paths)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=32768, min_frequency=8)
    tokenizer.train_from_iterator(data, trainer)
    tokenizer.post_processor = processors.ByteLevel()
    print("total number:", len(tokenizer.get_vocab()))
    tokenizer.save(tokenizer_path)
    


if __name__ == "__main__":
    # create_tokenizer()
    create_tokenizer()