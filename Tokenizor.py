import torch
from torch.utils.data import Dataset

class ClickbaitDataset(Dataset):
    def __init__(self, titles, bodies, labels, tokenizer, max_len=128):
        """
        titles: 뉴스 제목 리스트
        bodies: 뉴스 본문 리스트 (요약본 또는 앞부분)
        labels: 0(정상) 또는 1(낚시성/상이함)
        """
        self.titles = titles
        self.bodies = bodies
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        title = self.titles[idx]
        body = self.bodies[idx]
        label = self.labels[idx]

        # 제목 토큰화 (자동으로 Tensor로 변환 및 패딩)
        inputs_title = self.tokenizer.encode_plus(
            title,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 본문 토큰화
        inputs_body = self.tokenizer.encode_plus(
            body,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'title_ids': inputs_title['input_ids'].squeeze(0), # (Seq_len,)
            'body_ids': inputs_body['input_ids'].squeeze(0),   # (Seq_len,)
            'label': torch.tensor(label, dtype=torch.float32)  # (1,)
        }