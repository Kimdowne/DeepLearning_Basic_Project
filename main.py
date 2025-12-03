import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# 기존 파일에서 데이터 로더 함수 가져오기 (가정)
# from json2array import extract_value 
from kobert_tokenizer import KoBERTTokenizer

# === [설정] ===
DATA_PATH = "SampleData" # 데이터 경로
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 0.001
MAX_LEN = 64

import os
import json

def extract_value(path):

    newsTitles, newsContents, useTypes = [], [], []
    
    # 디렉토리가 실제로 존재하는지 확인
    if not os.path.exists(path):
        print(f"오류: '{path}' 경로를 찾을 수 없습니다.")
        return

    # 디렉토리 내의 파일 목록을 순회
    for filename in os.listdir(path):

        # .json 확인
        if filename.endswith(".json"):
            file_path = os.path.join(path, filename)
            
            try:
                # 파일 열기
                with open(file_path, 'r', encoding='utf-8') as f:
                    _data = json.load(f)

                    newsTitles.append(_data["sourceDataInfo"]["newsTitle"])

                    _content_list = [item.get("sentenceContent", "") for item in _data["sourceDataInfo"]["sentenceInfo"][:5]]
                    newsContents.append(" ".join(_content_list))
                    
                    useTypes.append(_data["sourceDataInfo"]["useType"])

                    
                    
            except json.JSONDecodeError:
                print(f"오류: {filename} 파일은 올바른 JSON 형식이 아닙니다.")
            except Exception as e:
                print(f"오류: {filename} 처리 중 문제 발생 - {e}")


    return newsTitles, newsContents, useTypes

def preprocess_data(titles, bodies, labels, tokenizer, max_len):
    title_ids = []
    body_ids = []
    
    for t, b in zip(titles, bodies):
        t_enc = tokenizer.encode_plus(t, add_special_tokens=True, max_length=max_len, 
                                      padding='max_length', truncation=True)
        b_enc = tokenizer.encode_plus(b, add_special_tokens=True, max_length=max_len, 
                                      padding='max_length', truncation=True)
        title_ids.append(t_enc['input_ids'])
        body_ids.append(b_enc['input_ids'])

    return (np.array(title_ids, dtype=np.int32), 
            np.array(body_ids, dtype=np.int32), 
            np.array(labels, dtype=np.float32))

def build_siamese_gru_model(vocab_size, max_len, embed_dim=128, hidden_dim=64):
    input_title = tf.keras.Input(shape=(max_len,), name='title_input')
    input_body = tf.keras.Input(shape=(max_len,), name='body_input')

    embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_dim)
    gru_layer = tf.keras.layers.GRU(hidden_dim)

    vec_title = gru_layer(embedding_layer(input_title))
    vec_body = gru_layer(embedding_layer(input_body))

    # 차분 층
    diff = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([vec_title, vec_body])

    x = tf.keras.layers.Dense(32, activation='relu')(diff)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=[input_title, input_body], outputs=output)


if __name__ == "__main__":

    target_path = Path("SampleData").resolve()
    raw_titles, raw_contents, raw_labels = extract_value(target_path)
    
    # 데이터 전처리 (Numpy 변환)
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
    X_title, X_body, y = preprocess_data(raw_titles, raw_contents, raw_labels, tokenizer, MAX_LEN)
    train_X = [X_title, X_body]

    # 모델 로드
    model = build_siamese_gru_model(vocab_size=tokenizer.vocab_size, max_len=MAX_LEN)
    
    # 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # 5. 학습 진행 (History 객체에 기록 저장)
    print("\n=== 학습 시작 ===")
    history = model.fit(
        x=train_X,  # 입력이 2개이므로 리스트로 전달
        y=y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2, # 검증 데이터 활용
        verbose=1
    )
    print("=== 학습 완료 ===")

    # 테스트 (예측)
    print("\n=== 테스트 수행 ===")
    test_title_raw = ["충격! 로또 1등 당첨 비결 공개"]
    test_body_raw = ["로또는 독립시행이므로 비결은 없습니다. 운에 맡기세요."]
    
    # 테스트 데이터 전처리
    X_test_t, X_test_b, _ = preprocess_data(test_title_raw, test_body_raw, [0], tokenizer, MAX_LEN)
    
    # 예측 수행
    prediction = model.predict([X_test_t, X_test_b])
    score = prediction[0][0]
    result = "낚시성 기사(불일치)" if score > 0.5 else "정상 기사(일치)"
    
    print(f"\n[테스트 결과]\n제목: {test_title_raw[0]}\n본문: {test_body_raw[0]}\n\n판정: {result} (점수: {score:.4f})")