import os
import csv
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights


# command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
parser.add_argument("--run-id", type=str, default=None)

args = parser.parse_args()

# hyperparameters
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
NUM_WORKERS = args.num_workers
OPTIMIZER_NAME = args.optimizer

if args.run_id is None:
    RUN_ID = f"resnet50_{OPTIMIZER_NAME}_lr{LR}_bs{BATCH_SIZE}_ep{EPOCHS}"
else:
    RUN_ID = args.run_id

device = "cuda"

print("device:", device)
print("run_id:", RUN_ID)
print("epochs:", EPOCHS)
print("batch_size:", BATCH_SIZE)
print("lr:", LR)
print("optimizer:", OPTIMIZER_NAME)


# project paths
PROJECT_ROOT = "/data/allen516/resnet50_cifar100"

TORCH_HOME = f"{PROJECT_ROOT}/cache/torch"
os.environ["TORCH_HOME"] = TORCH_HOME

DATA_ROOT = "/local_datasets/allen516/cifar100"
LOG_PATH = f"{PROJECT_ROOT}/logs/{RUN_ID}.csv"
CKPT_PATH = f"{PROJECT_ROOT}/checkpoints/{RUN_ID}_final.pth"

os.makedirs(TORCH_HOME, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

assert torch.cuda.is_available(), "CUDA GPU가 할당된 Slurm job 안에서 실행해야 합니다."


# csv log file
if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "epoch",
            "batch_size",
            "epochs",
            "lr",
            "optimizer",
            "train_loss",
            "train_acc",
            "test_acc",
        ])


# ImageNet pretrained model input setting normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# CIFAR-100 download/load
train_set = datasets.CIFAR100(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=train_tf,
)

test_set = datasets.CIFAR100(
    root=DATA_ROOT,
    train=False,
    download=True,
    transform=test_tf,
)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ImageNet pretrained ResNet-50
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# 100-class CIFAR-100 classifier
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)


# loss / optimizer
criterion = nn.CrossEntropyLoss()

if OPTIMIZER_NAME == "adam":
    optimizer = optim.Adam(model.parameters(), lr=LR)
else:
    optimizer = optim.AdamW(model.parameters(), lr=LR)


def train_one_epoch():
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # batch별로 loss를 더해줌
        correct += (outputs.argmax(1) == labels).sum().item()  # batch별로 맞춘 개수를 더해줌
        total += labels.size(0)

    loss = total_loss / len(train_loader)
    acc = correct / total * 100

    return loss, acc


def evaluate():
    model.eval()

    correct = 0  # 맞춘 개수
    total = 0  # 전체 이미지 개수

    with torch.no_grad():  # gradient 계산을 하지 않음
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100

    return acc


for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    test_acc = evaluate()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"loss: {train_loss:.4f} | "
        f"train acc: {train_acc:.2f}% | "
        f"test acc: {test_acc:.2f}%"
    )

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            RUN_ID,
            epoch + 1,
            BATCH_SIZE,
            EPOCHS,
            LR,
            OPTIMIZER_NAME,
            train_loss,
            train_acc,
            test_acc,
        ])


torch.save(model.state_dict(), CKPT_PATH)

print("saved checkpoint:", CKPT_PATH)
print("saved log:", LOG_PATH)