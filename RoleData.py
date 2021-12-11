import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import random

class RoleDataset(Dataset):
    def __init__(self, tokenizer, max_len, mode="song"):
        super(RoleDataset, self).__init__()
        self.mode = mode
        if self.mode == 'song':
            self.data = pd.read_csv('dataset/poetrySong/poetrySong.csv', sep='\t')
        elif self.mode == 'tang':
            self.data = pd.read_csv('dataset/poetryTang/poetryTang.csv', sep='\t')
        elif self.mode == 'ci':
            self.data = pd.read_csv('dataset/songci/songci.csv', sep='\t')
        else:
            self.data = pd.read_csv('dataset/CCPC/CCPC.csv', sep='\t')
        self.texts = self.data['content'].tolist()
        self.keywords = self.data['keywords'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.texts[index])

        if self.mode != "CCPC":
            encoding = self.tokenizer.encode_plus(text,
                                                  add_special_tokens=True,
                                                  max_length=self.max_len,
                                                  return_token_type_ids=True,
                                                  pad_to_max_length=True,
                                                  return_attention_mask=False,
                                                  return_tensors='pt')
            sample = {
                'texts': text,
                'input_ids': encoding['input_ids'].flatten()
            }
        else:
            keywords = str(self.keywords[index]).split(" ")
            keywords = keywords[random.randint(0, len(keywords) - 1):]
            random.shuffle(keywords)
            keywords = " ".join(keywords)
            keywords = keywords.replace(" ", "[SPACE]")  # '屏开[SPACE]晴日[SPACE]春风[SPACE]绿苔'

            encoding = self.tokenizer.encode_plus(keywords + " [SEP] " + text,
                                                  add_special_tokens=True,
                                                  max_length=self.max_len,
                                                  return_token_type_ids=True,
                                                  pad_to_max_length=True,
                                                  return_attention_mask=False,
                                                  return_tensors='pt')
            # '[CLS]屏开[SPACE]晴日[SPACE]春风[SPACE]绿苔[SEP]锄禾日当午[SEP]'
            # '[KEYWORD][KEYWORD][KEYWORD][KEYWORD][KEYWORD][KEYWORD][KEYWORD][KEYWORD][KEYWORD][KEYWORD][KEYWORD]
            # '[CONTENT][CONTENT][CONTENT][CONTENT][CONTENT][CONTENT]
            key_len = len(self.tokenizer.tokenize(keywords + " [SEP] ")) + 1  # CLS
            cnt_len = len(self.tokenizer.tokenize(text)) + 1  # SEP
            encoding_token = self.tokenizer.encode_plus("[KEYWORD]" * key_len + " [CONTENT] " * cnt_len,
                                                        add_special_tokens=False,
                                                        max_length=self.max_len,
                                                        return_token_type_ids=False,
                                                        pad_to_max_length=True,
                                                        return_attention_mask=False,
                                                        return_tensors='pt')
            sample = {
                'texts': text,
                'keywords': keywords,
                'input_ids': encoding['input_ids'].flatten(),
                'tokens': encoding_token['input_ids'].flatten()
            }

        return sample

    def __len__(self):
        return len(self.texts)


def create_dataloader(dataset, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
