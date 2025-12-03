import torch
import torch.nn as nn

class SiameseGRU(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super(SiameseGRU, self).__init__()
        
        # 1. 임베딩 층 (단어 인덱스 -> 벡터)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. GRU 층 (문맥 파악, 가중치 공유됨)
        # batch_first=True: 입력 형태가 (batch, seq, feature)임
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        
        # 3. 분류기 (차이 벡터를 입력받아 0~1 확률 출력)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # 0(유사) ~ 1(상이) 확률 출력
        )

    def forward_one(self, x):
        # x: (Batch, Seq_Len)
        embedded = self.embedding(x)      # (Batch, Seq, Embed_Dim)
        _, hidden = self.gru(embedded)    # hidden: (1, Batch, Hidden_Dim)
        return hidden[-1]                 # (Batch, Hidden_Dim) - 마지막 시점의 은닉 상태

    def forward(self, title_ids, body_ids):
        # 1. 두 입력을 동일한 GRU(forward_one)에 통과
        vector_title = self.forward_one(title_ids)
        vector_body = self.forward_one(body_ids)

        # 2. 두 벡터의 차이 계산 (L1 Distance: 절댓값 차이)
        # 낚시성 기사 탐지에는 '얼마나 다른가'가 중요하므로 차이값이 핵심 피쳐가 됨
        diff = torch.abs(vector_title - vector_body)

        # 3. 최종 판별
        output = self.fc(diff)
        return output