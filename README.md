# ResNet-50 CIFAR-100 Training

ImageNet 사전학습 ResNet-50을 CIFAR-100 데이터셋에 맞게 미세조정하는 PyTorch 학습 스크립트입니다.

> `resnet50_train.py`는 작성자가 제공한 원본을 수정하지 않고 그대로 올렸습니다.

## 스크립트 동작

- CIFAR-100 학습/테스트 데이터를 `./data`에 자동 다운로드합니다.
- 입력 이미지를 224 크기로 조정하고 ImageNet 정규화를 적용합니다.
- torchvision의 ImageNet 사전학습 ResNet-50을 불러옵니다.
- 마지막 분류기를 CIFAR-100용 100개 클래스로 교체합니다.
- 전체 모델을 Adam(`lr=1e-3`)으로 10 epoch 학습합니다.
- batch size는 64, DataLoader worker 수는 4입니다.
- epoch별 loss 및 정확도를 `training_log.csv`에 기록합니다.
- 최종 가중치를 `resnet50_cifar100.pth`로 저장합니다.

## RTX 3090 호환성 검토

대상 서버:

- NVIDIA GeForce RTX 3090
- Ampere, Compute Capability 8.6 (`sm_86`)
- VRAM 24 GB

검토 결과:

- 이 스크립트는 표준 PyTorch/torchvision 연산만 사용하므로 RTX 3090과 호환됩니다.
- CUDA 13.x 전용 의존성이 없습니다.
- Hopper/Ada 전용 커널 또는 Compute Capability 8.6 초과 요구사항이 없습니다.
- 별도 CUDA 확장, flash-attention, xformers, Triton 전용 커널을 사용하지 않습니다.
- ResNet-50, 224×224 입력, batch size 64, FP32, Adam 구성은 일반적으로 24 GB VRAM 안에서 실행 가능한 범위입니다.
- 다만 다른 프로세스가 GPU 메모리를 사용 중이면 OOM이 날 수 있으므로 실행 전에 `nvidia-smi`로 여유 메모리를 확인해야 합니다.
- 코드는 `device = "cuda"`로 고정되어 있으므로 CUDA GPU와 정상 설치된 NVIDIA 드라이버가 반드시 필요합니다.

## 권장 환경

재현성과 RTX 3090 지원을 위해 다음의 보수적인 조합을 권장합니다.

- Python 3.10
- PyTorch 2.5.1
- torchvision 0.20.1
- PyTorch CUDA wheel: CUDA 12.1 (`cu121`)

PyTorch wheel에 필요한 CUDA 런타임이 포함되므로, 일반적으로 서버에 별도 CUDA Toolkit을 설치할 필요는 없습니다. 호환되는 NVIDIA 드라이버는 필요합니다.

CUDA 11.x 환경이 필요한 서버에서는 같은 버전의 `cu118` wheel을 사용할 수 있습니다.

## 서버에서 설치 및 실행

```bash
git clone https://github.com/tydfuyhf/resnet50_train.git
cd resnet50_train

conda create -n resnet50-cifar100 python=3.10 -y
conda activate resnet50-cifar100

# 권장: CUDA 12.1 wheel
python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

CUDA 11.8 wheel을 사용해야 한다면 마지막 명령만 다음으로 바꿉니다.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

설치 확인:

```bash
nvidia-smi
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'none')"
```

정상이라면 GPU 이름에 RTX 3090, capability에 `(8, 6)`, `CUDA available`에 `True`가 표시됩니다.

학습 실행:

```bash
python resnet50_train.py
```

첫 실행에는 CIFAR-100 데이터와 ResNet-50 사전학습 가중치를 다운로드하므로 인터넷 연결이 필요합니다.

## 출력 파일

- `data/`: CIFAR-100 데이터
- `training_log.csv`: epoch별 학습 기록
- `resnet50_cifar100.pth`: 최종 모델 state dict

## 메모리 문제가 생길 경우

현재 원본 코드는 batch size 64와 FP32 학습을 사용합니다. RTX 3090 24 GB에서는 실행 가능할 것으로 예상되지만, 실제 VRAM 사용량은 드라이버, PyTorch 버전, 동시 GPU 프로세스에 따라 달라집니다.

OOM이 발생하면 다른 GPU 프로세스를 종료하거나, 원본 작성자의 판단에 따라 `BATCH_SIZE`를 32 또는 16으로 낮추는 방법이 있습니다. 이 저장소의 원본 스크립트에는 해당 변경을 적용하지 않았습니다.

## 참고

- [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)
- [PyTorch 이전 버전 설치 명령](https://pytorch.org/get-started/previous-versions/)
- [torchvision ResNet-50 문서](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
- [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)
