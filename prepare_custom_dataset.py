"""
커스텀 데이터셋 준비 스크립트
dataset_1, dataset_3, dataset_4를 합쳐서 train/val/test 생성
"""
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

# Raw dataset 경로
DATASET_PATHS = {
    "1": {
        "train": DATASET_DIR / "fake-news-classification" / "train.csv",
        "val": DATASET_DIR / "fake-news-classification" / "val.csv",
        "test": DATASET_DIR / "fake-news-classification" / "test.csv",
        "sep": ";",
        "index_col": 0
    },
    "3": {
        "train": DATASET_DIR / "fake-news-detection-datasets" / "train.csv",
        "test": DATASET_DIR / "fake-news-detection-datasets" / "test.csv",
        "sep": ",",
        "index_col": None
    },
    "4": {
        "train": DATASET_DIR / "llm-fake-news" / "train.csv",
        "test": DATASET_DIR / "llm-fake-news" / "test.csv",
        "sep": ",",
        "index_col": None
    }
}

# 라벨 정규화 매핑 (core/data.py와 동일)
LABEL_NORMALISATION = {
    "fake": 1,
    "false": 1,
    "fraud": 1,
    "real": 0,
    "true": 0,
    "legit": 0,
}


def normalise_label_value(value):
    """
    라벨 값을 0 또는 1로 정규화
    """
    # 숫자 타입
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(round(float(value)))
    
    # 결측치
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None  # 나중에 제거
    
    # 문자열 처리
    text = str(value).strip().lower()
    
    # 매핑 테이블 확인
    if text in LABEL_NORMALISATION:
        return LABEL_NORMALISATION[text]
    
    # 숫자 문자열
    if text.isdigit():
        return int(text)
    
    # f/t 약어
    if text in {"f", "t"}:
        return 1 if text == "f" else 0
    
    # 그 외는 그대로 반환 (경고 출력)
    logger.warning(f"Unknown label value: {value}")
    return value


def load_dataset(dataset_id: str, split: str, sample_ratio: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """
    데이터셋 로드 및 샘플링
    
    Args:
        dataset_id: 데이터셋 ID ('1', '3', '4')
        split: 'train', 'val', 'test'
        sample_ratio: 샘플링 비율 (0.0 ~ 1.0)
        seed: 랜덤 시드
    """
    config = DATASET_PATHS[dataset_id]
    
    if split not in config:
        logger.warning(f"Dataset {dataset_id} does not have {split} split")
        return pd.DataFrame()
    
    path = config[split]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    logger.info(f"Loading dataset {dataset_id} - {split}: {path.name}")
    
    # CSV 파일 읽기
    df = pd.read_csv(
        path,
        sep=config.get('sep', ','),
        index_col=config.get('index_col')
    )
    
    # 필수 컬럼 확인
    if 'label' not in df.columns:
        raise ValueError(f"{path.name} must contain 'label' column")
    if 'text' not in df.columns:
        raise ValueError(f"{path.name} must contain 'text' column")
    
    # title 컬럼이 없으면 추가
    if 'title' not in df.columns:
        logger.warning(f"{path.name} does not have 'title' column. Using empty string.")
        df['title'] = ""
    
    # 필요한 컬럼만 추출
    df = df[['title', 'text', 'label']].copy()
    
    # 결측치 처리
    df['title'] = df['title'].fillna("")
    df['text'] = df['text'].fillna("")
    
    # 라벨 정규화
    df['label'] = df['label'].apply(normalise_label_value)
    
    # 라벨이 None인 행 제거
    df = df.dropna(subset=['label'])
    
    # 라벨을 정수형으로 변환
    df['label'] = df['label'].astype(int)
    
    # 샘플링
    if sample_ratio < 1.0:
        original_len = len(df)
        df = df.sample(frac=sample_ratio, random_state=seed).reset_index(drop=True)
        logger.info(f"  - Sampled {len(df)}/{original_len} rows ({sample_ratio:.1%})")
    else:
        logger.info(f"  - Loaded {len(df)} rows")
    
    return df


def stratified_split(df: pd.DataFrame, train_ratio: float, seed: int = 42):
    """
    Stratified train/val split (레이블 비율 유지)
    """
    np.random.seed(seed)
    
    train_dfs = []
    val_dfs = []
    
    # 각 레이블별로 split
    for label in df['label'].unique():
        label_df = df[df['label'] == label].copy()
        label_df = label_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        split_idx = int(len(label_df) * train_ratio)
        train_dfs.append(label_df[:split_idx])
        val_dfs.append(label_df[split_idx:])
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    
    # Shuffle
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return train_df, val_df


def prepare_train_val_data(
    dataset_3_ratio: float = 0.1,
    dataset_4_ratio: float = 0.01,
    train_val_ratio: float = 0.8,
    seed: int = 42
):
    """
    Train/Val 데이터 준비
    
    Args:
        dataset_3_ratio: dataset_3 샘플링 비율
        dataset_4_ratio: dataset_4 샘플링 비율
        train_val_ratio: train/val split 비율
        seed: 랜덤 시드
    """
    logger.info("\n=== Preparing Train/Val Data ===")
    logger.info(f"Dataset 3 sample ratio: {dataset_3_ratio:.1%}")
    logger.info(f"Dataset 4 sample ratio: {dataset_4_ratio:.1%}")
    logger.info(f"Train/Val split ratio: {train_val_ratio:.2f}/{1-train_val_ratio:.2f}")
    
    # Dataset 1: train + val 합치기
    df1_train = load_dataset("1", "train", sample_ratio=1.0, seed=seed)
    df1_val = load_dataset("1", "val", sample_ratio=1.0, seed=seed)
    df1 = pd.concat([df1_train, df1_val], ignore_index=True)
    logger.info(f"Dataset 1 total: {len(df1)} rows")
    
    # Dataset 3: train만 사용 (샘플링)
    df3 = load_dataset("3", "train", sample_ratio=dataset_3_ratio, seed=seed)
    
    # Dataset 4: train만 사용 (샘플링)
    df4 = load_dataset("4", "train", sample_ratio=dataset_4_ratio, seed=seed)
    
    # 모든 데이터 합치기
    combined_df = pd.concat([df1, df3, df4], ignore_index=True)
    logger.info(f"\nCombined total: {len(combined_df)} rows")
    
    # Label 분포 확인
    label_dist = combined_df['label'].value_counts()
    logger.info(f"Label distribution:")
    for label, count in label_dist.items():
        logger.info(f"  - {label}: {count} ({count/len(combined_df)*100:.1f}%)")
    
    # Train/Val split (stratified)
    train_df, val_df = stratified_split(combined_df, train_val_ratio, seed)
    
    logger.info(f"\nSplit results:")
    logger.info(f"  - Train: {len(train_df)} rows")
    logger.info(f"  - Val: {len(val_df)} rows")
    
    # Split 후 레이블 분포 확인
    train_label_dist = train_df['label'].value_counts()
    val_label_dist = val_df['label'].value_counts()
    
    logger.info(f"\nTrain label distribution:")
    for label, count in train_label_dist.items():
        logger.info(f"  - {label}: {count} ({count/len(train_df)*100:.1f}%)")
    
    logger.info(f"\nVal label distribution:")
    for label, count in val_label_dist.items():
        logger.info(f"  - {label}: {count} ({count/len(val_df)*100:.1f}%)")
    
    return train_df, val_df


def prepare_test_data() -> pd.DataFrame:
    """
    Test 데이터 준비 (dataset 1, 3, 4의 test 합치기)
    """
    logger.info("\n=== Preparing Test Data ===")
    
    # 각 데이터셋의 test 로드
    df1_test = load_dataset("1", "test", sample_ratio=1.0)
    df3_test = load_dataset("3", "test", sample_ratio=1.0)
    df4_test = load_dataset("4", "test", sample_ratio=1.0)
    
    # 합치기
    test_df = pd.concat([df1_test, df3_test, df4_test], ignore_index=True)
    logger.info(f"\nTest total: {len(test_df)} rows")
    
    # Label 분포 확인
    label_dist = test_df['label'].value_counts()
    logger.info(f"Test label distribution:")
    for label, count in label_dist.items():
        logger.info(f"  - {label}: {count} ({count/len(test_df)*100:.1f}%)")
    
    return test_df


def save_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_name: str = "custom-dataset"
) -> Path:
    """
    데이터셋을 CSV 파일로 저장
    
    Args:
        train_df: Train 데이터프레임
        val_df: Validation 데이터프레임
        test_df: Test 데이터프레임
        output_name: 출력 디렉토리 이름
    
    Returns:
        출력 디렉토리 경로
    """
    output_dir = DATASET_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    
    # CSV 파일 저장
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"\n=== Datasets Saved ===")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  - Train: {train_path.name} ({len(train_df)} rows)")
    logger.info(f"  - Val: {val_path.name} ({len(val_df)} rows)")
    logger.info(f"  - Test: {test_path.name} ({len(test_df)} rows)")
    
    return output_dir


def main():
    """CLI 인터페이스"""
    parser = argparse.ArgumentParser(
        description="Prepare custom dataset by merging dataset 1, 3, 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 설정으로 실행
  python prepare_custom_dataset.py
  
  # 샘플링 비율 변경
  python prepare_custom_dataset.py --dataset3-ratio 0.2 --dataset4-ratio 0.05
  
  # Train/Val 비율 변경
  python prepare_custom_dataset.py --train-val-ratio 0.9
  
  # 모든 하이퍼파라미터 지정
  python prepare_custom_dataset.py --dataset3-ratio 0.15 --dataset4-ratio 0.02 --train-val-ratio 0.85 --seed 123
        """
    )
    
    parser.add_argument(
        "--dataset3-ratio",
        type=float,
        default=0.1,
        help="Dataset 3 sampling ratio (default: 0.1 = 10%%)"
    )
    
    parser.add_argument(
        "--dataset4-ratio",
        type=float,
        default=0.01,
        help="Dataset 4 sampling ratio (default: 0.01 = 1%%)"
    )
    
    parser.add_argument(
        "--train-val-ratio",
        type=float,
        default=0.8,
        help="Train/Val split ratio (default: 0.8 = 80%% train, 20%% val)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--output-name",
        type=str,
        default="custom-dataset",
        help="Output directory name (default: custom-dataset)"
    )
    
    args = parser.parse_args()
    
    try:
        # 파라미터 검증
        if not 0.0 < args.dataset3_ratio <= 1.0:
            raise ValueError("dataset3-ratio must be between 0 and 1")
        if not 0.0 < args.dataset4_ratio <= 1.0:
            raise ValueError("dataset4-ratio must be between 0 and 1")
        if not 0.0 < args.train_val_ratio < 1.0:
            raise ValueError("train-val-ratio must be between 0 and 1")
        
        # Train/Val 데이터 준비
        train_df, val_df = prepare_train_val_data(
            dataset_3_ratio=args.dataset3_ratio,
            dataset_4_ratio=args.dataset4_ratio,
            train_val_ratio=args.train_val_ratio,
            seed=args.seed
        )
        
        # Test 데이터 준비
        test_df = prepare_test_data()
        
        # 파일 저장
        output_dir = save_datasets(
            train_df, val_df, test_df,
            output_name=args.output_name
        )
        
        logger.info("\n✓ Dataset preparation completed successfully!")
        logger.info(f"\nTo use this dataset, run:")
        logger.info(f"  DATASET_NAME = '{args.output_name}'")
        
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

