from pathlib import Path

from json2array import extract_value
from Tokenizor import ClickbaitDataset
from GRU import SiameseGRU

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

from kobert_tokenizer import KoBERTTokenizer

# Raw Data 수집
DATA_PATH = "SampleData"
target_path = Path(__file__).resolve().parent / DATA_PATH

if __name__ == "__main__":

    # Parameter
    BATCH_SIZE = 2
    EPOCHS = 5
    LEARNING_RATE = 0.001
    MAX_LEN = 64

    # json에서 내용 추출
    raw_titles, raw_contents, raw_labels = extract_value(target_path)
    
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
    
    # 데이터셋 & 데이터로더 생성
    dataset = ClickbaitDataset(raw_titles, raw_contents, raw_labels, tokenizer, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseGRU(vocab_size=tokenizer.vocab_size, embed_dim=128, hidden_dim=64).to(device)
    
    # Torch Train
    
    # 손실함수 및 최적화 도구
    criterion = nn.BCELoss() # 이진 분류 (0 or 1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("=== 학습 시작 ===")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0

        for batch in dataloader:

            # 데이터 로드
            title_ids = batch['title_ids'].to(device)
            body_ids = batch['body_ids'].to(device)
            labels = batch['label'].to(device).unsqueeze(1) # (Batch, 1) 형태로 맞춤

            # 1. Forward
            outputs = model(title_ids, body_ids)
            
            # 2. Loss 계산
            loss = criterion(outputs, labels)
            
            # 3. Backward & Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

    print("=== 학습 완료 ===")

    # --- 테스트 (예측) ---
    model.eval()
    with torch.no_grad():
        test_title = ["충격! 로또 1등 당첨 비결 공개"]
        test_body = ["로또는 독립시행이므로 비결은 없습니다. 운에 맡기세요."]
        
        # 추론을 위한 입력 변환
        inp_t = tokenizer.batch_encode_plus(test_title, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')['input_ids'].to(device)
        inp_b = tokenizer.batch_encode_plus(test_body, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')['input_ids'].to(device)
        
        prediction = model(inp_t, inp_b)
        result = "낚시성 기사(불일치)" if prediction.item() > 0.5 else "정상 기사(일치)"
        print(f"\n[테스트 결과]\n제목: {test_title[0]}\n판정: {result} (점수: {prediction.item():.4f})")