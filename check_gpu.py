#!/usr/bin/env python3
"""
GPU 확인 스크립트
CUDA가 사용 가능한지, 그리고 학습이 GPU에서 실행되는지 확인합니다.
"""
import torch
import sys

def check_gpu():
    """GPU 상태 확인"""
    print("=" * 70)
    print("🖥️  GPU 상태 확인")
    print("=" * 70)
    
    # PyTorch 버전
    print(f"\n📦 PyTorch 버전: {torch.__version__}")
    
    # CUDA 사용 가능 여부
    cuda_available = torch.cuda.is_available()
    print(f"\n🔍 CUDA 사용 가능: {'✅ Yes' if cuda_available else '❌ No'}")
    
    if cuda_available:
        # GPU 정보
        print(f"\n💻 GPU 정보:")
        print(f"   - GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"     - 이름: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"     - 총 메모리: {props.total_memory / 1024**3:.2f} GB")
            print(f"     - Compute Capability: {props.major}.{props.minor}")
        
        # 현재 GPU
        print(f"\n🎯 현재 사용 GPU: cuda:{torch.cuda.current_device()}")
        
        # 간단한 연산 테스트
        print(f"\n🧪 GPU 연산 테스트:")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print(f"   ✅ GPU 연산 성공! (행렬 곱셈 테스트 통과)")
            print(f"   - 사용 디바이스: {z.device}")
            
            # 메모리 사용량
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            cached = torch.cuda.memory_reserved(0) / 1024**2
            print(f"   - 할당된 메모리: {allocated:.2f} MB")
            print(f"   - 캐시된 메모리: {cached:.2f} MB")
            
        except Exception as e:
            print(f"   ❌ GPU 연산 실패: {e}")
            return False
        
        print(f"\n✨ 학습 시 자동으로 GPU가 사용됩니다!")
        return True
    
    else:
        print(f"\n❌ CUDA를 사용할 수 없습니다.")
        print(f"\n💡 GPU를 사용하려면:")
        print(f"   1. NVIDIA GPU가 시스템에 설치되어 있는지 확인")
        print(f"   2. NVIDIA 드라이버 설치")
        print(f"   3. CUDA Toolkit 설치")
        print(f"   4. PyTorch CUDA 버전 재설치:")
        print(f"      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print(f"\n⚠️  현재는 CPU로 학습됩니다 (속도가 느릴 수 있습니다)")
        return False
    
    print("=" * 70)

def check_cudnn():
    """cuDNN 상태 확인"""
    if torch.cuda.is_available():
        print(f"\n🔧 cuDNN:")
        print(f"   - cuDNN 사용 가능: {'✅ Yes' if torch.backends.cudnn.is_available() else '❌ No'}")
        if torch.backends.cudnn.is_available():
            print(f"   - cuDNN 버전: {torch.backends.cudnn.version()}")
            print(f"   - cuDNN Enabled: {torch.backends.cudnn.enabled}")

if __name__ == "__main__":
    print("\n")
    success = check_gpu()
    check_cudnn()
    print("\n")
    
    # 종료 코드
    sys.exit(0 if success else 1)

