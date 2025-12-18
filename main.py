import os
import glob
import math
import numpy as np
import tensorflow as tf
import json
from kobert_tokenizer import KoBERTTokenizer

# ==============================================================================
# [설정] 환경 변수 및 GPU 설정
# ==============================================================================

# 1. 토크나이저 병렬 처리 비활성화 (필수: 교착 상태 방지)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. GPU 메모리 동적 할당 (OOM 방지)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 가속 활성화됨: {len(gpus)}개의 GPU 감지됨")
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")
else:
    print("⚠️ GPU를 찾을 수 없습니다. CPU로 학습합니다.")

# ==============================================================================

# === [파라미터 설정] ===
BATCH_SIZE = 32  
EPOCHS = 100
LEARNING_RATE = 0.001
MAX_LEN = 64

# === [1. 데이터 제너레이터] ===
def data_generator(file_paths):
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                info = data.get("sourceDataInfo", {})
                title = info.get("newsTitle", "")
                
                # 본문 앞 5문장 추출
                sentences = info.get("sentenceInfo", [])
                content_list = [item.get("sentenceContent", "") for item in sentences[:5]]
                body = " ".join(content_list)
                
                # 라벨
                label_val = info.get("useType", 0)
                label = float(label_val)
                
                yield title, body, label

        except Exception as e:
            continue

# === [2. 데이터셋 파이프라인 생성 함수] ===
def create_dataset(data_path, tokenizer, max_len, batch_size):
    # 1. 파일 탐색 (재귀적)
    print(f"📂 경로 탐색 중: {data_path}")
    search_pattern = os.path.join(data_path, "**", "*.json")
    all_files = glob.glob(search_pattern, recursive=True)
    
    file_count = len(all_files)
    print(f"   ㄴ 발견된 파일 수: {file_count}개")
    
    if file_count == 0:
        return None, 0

    # 2. 제너레이터 연결
    def gen():
        yield from data_generator(all_files)

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )

    # 3. 토크나이징 및 매핑
    def tokenize_map(title, body, label):
        def py_tokenize(t, b):
            t_str = t.numpy().decode('utf-8')
            b_str = b.numpy().decode('utf-8')
            
            t_enc = tokenizer.encode_plus(t_str, max_length=max_len, padding='max_length', truncation=True)
            b_enc = tokenizer.encode_plus(b_str, max_length=max_len, padding='max_length', truncation=True)
            
            return np.array(t_enc['input_ids'], dtype=np.int32), np.array(b_enc['input_ids'], dtype=np.int32)

        title_ids, body_ids = tf.py_function(py_tokenize, [title, body], [tf.int32, tf.int32])
        
        title_ids.set_shape([max_len])
        body_ids.set_shape([max_len])
        
        return ({"title_input": title_ids, "body_input": body_ids}, label)

    # 병렬 처리 설정
    dataset = dataset.map(tokenize_map, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, file_count

# === [3. 모델 구조 정의 (Siamese GRU)] ===
def build_siamese_gru_model(vocab_size, max_len, embed_dim=128, hidden_dim=64):
    input_title = tf.keras.Input(shape=(max_len,), name='title_input')
    input_body = tf.keras.Input(shape=(max_len,), name='body_input')

    embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)
    gru_layer = tf.keras.layers.GRU(hidden_dim)

    vec_title = gru_layer(embedding_layer(input_title))
    vec_body = gru_layer(embedding_layer(input_body))

    # 차이 벡터 계산 (L1 Distance)
    diff = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([vec_title, vec_body])

    x = tf.keras.layers.Dense(32, activation='relu')(diff)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=[input_title, input_body], outputs=output)

# === [4. 메인 실행 블록] ===
if __name__ == "__main__":
    
    print("\n=== 프로그램 시작 ===")
    
    # 1. 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_PATH = os.path.join(BASE_DIR, "DataSet", "train")
    TEST_PATH = os.path.join(BASE_DIR, "DataSet", "test")

    # 2. 토크나이저 로드
    try:
        tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
    except Exception as e:
        print(f"❌ 토크나이저 로드 실패: {e}")
        exit()

    # 3. [학습] 데이터셋 생성
    print("\n--- [Train] 데이터셋 준비 ---")
    if not os.path.exists(TRAIN_PATH):
        print(f"❌ 학습 데이터 폴더 없음: {TRAIN_PATH}")
        exit()
        
    train_ds, train_files = create_dataset(TRAIN_PATH, tokenizer, MAX_LEN, BATCH_SIZE)
    
    if train_ds is None:
        print("❌ 학습 데이터가 없습니다.")
        exit()

    train_steps = math.ceil(train_files / BATCH_SIZE)
    print(f"   ㄴ 학습 스텝 수: {train_steps} (총 {train_files}개)")

    # 4. 모델 생성 및 컴파일
    print("\n--- 모델 빌드 ---")
    model = build_siamese_gru_model(vocab_size=tokenizer.vocab_size, max_len=MAX_LEN)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 5. 모델 학습
    print("\n=== 학습 시작 ===")
    try:
        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            steps_per_epoch=train_steps,
            verbose=1
        )
        print("=== 학습 완료 ===")
        
        # [수정됨] .keras 포맷으로 저장 (Warning 해결)
        save_name = "siamese_gru_model.keras"
        model.save(save_name)
        print(f"💾 모델 저장 완료: {save_name}")
        
    except KeyboardInterrupt:
        print("\n⛔ 사용자에 의해 학습이 중단되었습니다.")
        exit()
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        exit()

    # 6. [평가] 테스트 데이터셋으로 성능 평가
    print("\n=== 모델 평가 (Test) ===")
    
    if not os.path.exists(TEST_PATH):
        print(f"⚠️ 테스트 폴더가 없어 평가를 건너뜁니다: {TEST_PATH}")
    else:
        print(f"--- [Test] 데이터셋 준비 ---")
        test_ds, test_files = create_dataset(TEST_PATH, tokenizer, MAX_LEN, BATCH_SIZE)
        
        if test_ds is not None:
            test_steps = math.ceil(test_files / BATCH_SIZE)
            print(f"   ㄴ 테스트 스텝 수: {test_steps} (총 {test_files}개)")
            
            print("\n--- 평가 진행 중... ---")
            # evaluate 함수로 손실과 정확도 계산
            test_loss, test_acc = model.evaluate(test_ds, steps=test_steps, verbose=1)
            
            print("\n📊 [최종 평가 결과]")
            print(f"   - Test Loss    : {test_loss:.4f}")
            print(f"   - Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        else:
            print("⚠️ 테스트 데이터 파일(.json)을 찾을 수 없습니다.")