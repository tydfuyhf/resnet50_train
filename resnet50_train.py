import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

# hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 4
RUN_ID = "resnet50_adam_lr1e-3_bs64"
device = "cuda"
print("device:", device)

# project paths
PROJECT_ROOT = "/data/allen516/resnet50_cifar100"
TORCH_HOME = f"{PROJECT_ROOT}/cache/torch"
os.environ["TORCH_HOME"] = TORCH_HOME
os.makedirs(TORCH_HOME, exist_ok=True)
DATA_ROOT = "/local_datasets/allen516/cifar100"
LOG_PATH = f"{PROJECT_ROOT}/logs/training_log.csv" # csv file to save training log
CKPT_PATH = f"{PROJECT_ROOT}/checkpoints/resnet50_cifar100.pth"

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

assert torch.cuda.is_available(), "CUDA GPU가 할당된 Slurm job 안에서 실행해야 합니다."

if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "epoch", "batch_size", "epochs", "lr", "optimizer", "train_loss", "train_acc", "test_acc"])

# ImageNet pretrained model input setting normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize(mean, std),])
test_tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std), ])

# CIFAR-100 download/load
train_set = datasets.CIFAR100( root=DATA_ROOT, train=True, download=True, transform=train_tf, )
test_set = datasets.CIFAR100( root=DATA_ROOT, train=False, download=True, transform=test_tf, )

train_loader = DataLoader( train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, )
test_loader = DataLoader( test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, )

# ImageNet pretrained ResNet-50
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# 100-class CIFAR-100 classifier
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# loss / optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

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

        total_loss += loss.item() # batch별로 loss를 더해줌
        correct += (outputs.argmax(1) == labels).sum().item() # batch별로 맞춘 개수를 더해줌
        total += labels.size(0)
        
    loss = total_loss / len(train_loader)
    acc = correct / total * 100
    return loss, acc

def evaluate():
    model.eval()
    correct = 0 # 맞춘 개수
    total = 0 # 전체 이미지 개수
    with torch.no_grad(): # gradient 계산을 하지 않음
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

    print(f"Epoch {epoch + 1}/{EPOCHS} | loss: {train_loss:.4f} | train acc: {train_acc:.2f}% | test acc: {test_acc:.2f}%")

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([RUN_ID, epoch + 1, BATCH_SIZE, EPOCHS, LR, "Adam", train_loss, train_acc, test_acc])

torch.save(model.state_dict(), CKPT_PATH)
print("saved:", CKPT_PATH)
print("saved:", LOG_PATH)