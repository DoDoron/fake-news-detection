# Custom Dataset 사용 가이드

Dataset 1, 3, 4를 합쳐서 커스텀 데이터셋을 만들고 학습하는 방법을 설명합니다.

## 📋 개요

이 시스템은 다음과 같은 방식으로 데이터를 구성합니다:

### Train 데이터 구성
- **Dataset 1**: train.csv + val.csv (전체 사용)
- **Dataset 3**: train.csv (샘플링 비율 조정 가능)
- **Dataset 4**: train.csv (샘플링 비율 조정 가능)

### Validation 데이터 구성
- 위의 merged train 데이터를 train/val로 split (비율 조정 가능)

### Test 데이터 구성
- **Dataset 1**: test.csv (전체 사용)
- **Dataset 3**: test.csv (전체 사용)
- **Dataset 4**: test.csv (전체 사용)

## 🎯 하이퍼파라미터

다음 3가지 하이퍼파라미터를 조정할 수 있습니다:

1. **`--dataset3-ratio`**: Dataset 3의 샘플링 비율 (기본값: 0.1 = 10%)
2. **`--dataset4-ratio`**: Dataset 4의 샘플링 비율 (기본값: 0.01 = 1%)
3. **`--train-val-ratio`**: Train/Val split 비율 (기본값: 0.8 = 80% train, 20% val)

## 🚀 사용 방법

### 방법 1: 명령줄에서 실행

```bash
# 기본 설정으로 실행
python prepare_custom_dataset.py

# 샘플링 비율 변경
python prepare_custom_dataset.py --dataset3-ratio 0.2 --dataset4-ratio 0.05

# Train/Val 비율 변경 (90:10)
python prepare_custom_dataset.py --train-val-ratio 0.9

# 모든 파라미터 지정
python prepare_custom_dataset.py \
    --dataset3-ratio 0.15 \
    --dataset4-ratio 0.02 \
    --train-val-ratio 0.85 \
    --seed 123 \
    --output-name my-custom-dataset
```

### 방법 2: 노트북에서 실행

`notebooks/train_custom_dataset.ipynb` 파일을 사용하세요.

```python
# 하이퍼파라미터 설정
DATASET_3_RATIO = 0.1
DATASET_4_RATIO = 0.01
TRAIN_VAL_RATIO = 0.8
RANDOM_SEED = 42
OUTPUT_DATASET_NAME = "custom-dataset"

# 데이터셋 준비 (셸 명령어로 실행)
!cd .. && python prepare_custom_dataset.py \
    --dataset3-ratio {DATASET_3_RATIO} \
    --dataset4-ratio {DATASET_4_RATIO} \
    --train-val-ratio {TRAIN_VAL_RATIO} \
    --seed {RANDOM_SEED} \
    --output-name {OUTPUT_DATASET_NAME}

# 데이터 로드 및 학습
loaders, vocab, tokenizer, info = build_datasets(
    name=OUTPUT_DATASET_NAME,
    batch_size=64,
    max_len=512,
    num_workers=4,
    max_vocab_size=20000,
)
```

## 📊 실험 예시

### 실험 1: 샘플링 비율 변화

```bash
# 적은 샘플링
python prepare_custom_dataset.py \
    --dataset3-ratio 0.05 \
    --dataset4-ratio 0.005 \
    --output-name custom-small

# 중간 샘플링 (기본값)
python prepare_custom_dataset.py \
    --dataset3-ratio 0.1 \
    --dataset4-ratio 0.01 \
    --output-name custom-medium

# 많은 샘플링
python prepare_custom_dataset.py \
    --dataset3-ratio 0.2 \
    --dataset4-ratio 0.02 \
    --output-name custom-large
```

### 실험 2: Train/Val 비율 변화

```bash
# 70:30 split
python prepare_custom_dataset.py --train-val-ratio 0.7 --output-name custom-70-30

# 80:20 split (기본값)
python prepare_custom_dataset.py --train-val-ratio 0.8 --output-name custom-80-20

# 90:10 split
python prepare_custom_dataset.py --train-val-ratio 0.9 --output-name custom-90-10
```

## 📁 출력 구조

실행하면 다음과 같은 구조로 데이터셋이 생성됩니다:

```
dataset/
  custom-dataset/          # 또는 지정한 output-name
    ├── train.csv          # Train 데이터
    ├── val.csv            # Validation 데이터
    └── test.csv           # Test 데이터
```

각 CSV 파일은 다음 컬럼을 포함합니다:
- `title`: 기사 제목
- `text`: 기사 본문
- `label`: 레이블 (0: real, 1: fake)

## 🔍 데이터 통계 확인

스크립트 실행 시 다음 정보가 출력됩니다:

```
=== Preparing Train/Val Data ===
Dataset 3 sample ratio: 10.0%
Dataset 4 sample ratio: 1.0%
Train/Val split ratio: 0.80/0.20

Loading dataset 1 - train: train.csv
  - Loaded 26938 rows
Loading dataset 1 - val: val.csv
  - Loaded 8979 rows
Dataset 1 total: 35917 rows

Loading dataset 3 - train: train.csv
  - Sampled 2694/26938 rows (10.0%)

Loading dataset 4 - train: train.csv
  - Sampled 2320/232003 rows (1.0%)

Combined total: 40931 rows
Label distribution:
  - 0: 20465 (50.0%)
  - 1: 20466 (50.0%)

Split results:
  - Train: 32745 rows
  - Val: 8186 rows
```

## 🎓 학습 실행

데이터셋이 생성되면 기존 방식대로 학습할 수 있습니다:

```python
from core.data import build_datasets
from core.train_eval import train_and_evaluate

# 데이터 로드
loaders, vocab, tokenizer, info = build_datasets(
    name='custom-dataset',
    batch_size=64,
    max_len=512,
    num_workers=4,
    max_vocab_size=20000,
)

# 모델 학습
model = MODEL_REGISTRY['bilstm'](vocab_size=len(vocab), num_classes=2)
results, run_dir = train_and_evaluate(
    model, loaders, config,
    dataset_name='custom-dataset',
    model_name='bilstm',
)
```

## 💡 팁

1. **처음 실험**: 작은 샘플링 비율로 시작해서 빠르게 테스트
   ```bash
   python prepare_custom_dataset.py --dataset3-ratio 0.01 --dataset4-ratio 0.001
   ```

2. **재현성**: 동일한 결과를 위해 `--seed` 값을 고정
   ```bash
   python prepare_custom_dataset.py --seed 42
   ```

3. **클래스 불균형**: 스크립트가 자동으로 stratified split을 수행하여 train/val의 레이블 비율을 동일하게 유지합니다.

4. **메모리 부족**: Dataset 3, 4의 샘플링 비율을 낮추세요.

## 🐛 문제 해결

### 에러: "Dataset file not found"
- 데이터셋 파일이 올바른 경로에 있는지 확인하세요
- 경로: `dataset/fake-news-classification/`, `dataset/fake-news-detection-datasets/`, `dataset/llm-fake-news/`

### 에러: "must contain 'label' column"
- CSV 파일에 `label` 컬럼이 있는지 확인하세요

### 메모리 부족
- 샘플링 비율을 낮추세요
- 또는 batch_size를 줄이세요

## 📝 참고사항

- Train/Val split 시 **stratified sampling**을 사용하여 레이블 비율을 유지합니다
- Test 데이터는 샘플링하지 않고 전체를 사용합니다
- 모든 데이터는 `title`, `text`, `label` 컬럼으로 정규화됩니다
- 결측치는 자동으로 처리됩니다 (title/text는 빈 문자열로, label 없는 행은 제거)

