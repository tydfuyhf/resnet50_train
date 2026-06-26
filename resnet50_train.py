import os
import csv
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights


# command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--mode", choices=["tune", "final"], default="tune")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
parser.add_argument("--weight-decay", type=float, default=None)
parser.add_argument("--run-id", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

# hyperparameters
MODE = args.mode
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
NUM_WORKERS = args.num_workers
OPTIMIZER_NAME = args.optimizer
SEED = args.seed

# Adam baseline은 weight decay 0, AdamW 기본값은 1e-4
if args.weight_decay is None:
    WEIGHT_DECAY = 1e-4 if OPTIMIZER_NAME == "adamw" else 0.0
else:
    WEIGHT_DECAY = args.weight_decay

if args.run_id is None:
    RUN_ID = (
        f"resnet50_{MODE}_{OPTIMIZER_NAME}"
        f"_lr{LR:.0e}"
        f"_bs{BATCH_SIZE}"
        f"_ep{EPOCHS}"
        f"_wd{WEIGHT_DECAY:.0e}"
        f"_seed{SEED}"
    )
else:
    RUN_ID = args.run_id

device = "cuda"

print("device:", device)
print("mode:", MODE)
print("run_id:", RUN_ID)
print("epochs:", EPOCHS)
print("batch_size:", BATCH_SIZE)
print("lr:", LR)
print("optimizer:", OPTIMIZER_NAME)
print("weight_decay:", WEIGHT_DECAY)
print("seed:", SEED)


# project paths
PROJECT_ROOT = "/data/allen516/resnet50_cifar100"

TORCH_HOME = f"{PROJECT_ROOT}/cache/torch"
os.environ["TORCH_HOME"] = TORCH_HOME

DATA_ROOT = "/local_datasets/allen516/cifar100"
LOG_PATH = f"{PROJECT_ROOT}/logs/{RUN_ID}.csv"

if MODE == "tune":
    CKPT_PATH = f"{PROJECT_ROOT}/checkpoints/{RUN_ID}_best_val.pth"
else:
    CKPT_PATH = f"{PROJECT_ROOT}/checkpoints/{RUN_ID}_final.pth"

os.makedirs(TORCH_HOME, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

assert torch.cuda.is_available(), "CUDA GPU가 할당된 Slurm job 안에서 실행해야 합니다."

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# csv log file
if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "mode",
            "epoch",
            "batch_size",
            "epochs",
            "lr",
            "optimizer",
            "weight_decay",
            "seed",
            "train_loss",
            "train_acc",
            "val_acc",
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

eval_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# CIFAR-100 download/load
# 같은 CIFAR-100 train data지만 transform만 다르게 준비
train_dataset_for_split = datasets.CIFAR100(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=train_tf,
)

eval_dataset_for_split = datasets.CIFAR100(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=eval_tf,
)

test_set = datasets.CIFAR100(
    root=DATA_ROOT,
    train=False,
    download=True,
    transform=eval_tf,
)

# tune / final 모두 마지막에 test accuracy를 기록하므로 항상 생성
test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# tune mode:
# 50,000 train images -> 45,000 train + 5,000 validation
# final mode:
# 선택된 hyperparameter로 전체 50,000 train images 사용
if MODE == "tune":
    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(train_dataset_for_split),
        generator=generator,
    ).tolist()

    train_indices = indices[:45000]
    val_indices = indices[45000:]

    train_set = Subset(train_dataset_for_split, train_indices)
    val_set = Subset(eval_dataset_for_split, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print("train samples:", len(train_set))
    print("validation samples:", len(val_set))

else:
    train_set = train_dataset_for_split

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print("train samples:", len(train_set))
    print("test samples:", len(test_set))


# ImageNet pretrained ResNet-50
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# 100-class CIFAR-100 classifier
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)


# loss / optimizer
criterion = nn.CrossEntropyLoss()

if OPTIMIZER_NAME == "adam":
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
else:
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )


def train_one_epoch():
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    loss = total_loss / len(train_loader)
    acc = correct / total * 100

    return loss, acc


def evaluate(loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100

    return acc


# tune mode에서는 validation accuracy가 최고인 checkpoint 저장
best_val_acc = -1.0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()

    if MODE == "tune":
        val_acc = evaluate(val_loader)
        test_acc = evaluate(test_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_PATH)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.2f}% | "
            f"val acc: {val_acc:.2f}% | "
            f"test acc: {test_acc:.2f}%"
        )

    else:
        val_acc = ""
        test_acc = evaluate(test_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.2f}%"
            f"test acc: {test_acc:.2f}%"
        )

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            RUN_ID,
            MODE,
            epoch + 1,
            BATCH_SIZE,
            EPOCHS,
            LR,
            OPTIMIZER_NAME,
            WEIGHT_DECAY,
            SEED,
            train_loss,
            train_acc,
            val_acc,
            test_acc,
        ])


# tune:
# best validation checkpoint 기준 test accuracy 기록
if MODE == "tune":
    model.load_state_dict(
        torch.load(
            CKPT_PATH,
            map_location=device,
            weights_only=True,
        )
    )

    test_acc = evaluate(test_loader)

    print(f"best validation acc: {best_val_acc:.2f}%")
    print(f"test acc of best validation checkpoint: {test_acc:.2f}%")

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            RUN_ID,
            MODE,
            "best_val_test",
            BATCH_SIZE,
            EPOCHS,
            LR,
            OPTIMIZER_NAME,
            WEIGHT_DECAY,
            SEED,
            "",
            "",
            best_val_acc,
            test_acc,
        ])

    print("saved best validation checkpoint:", CKPT_PATH)
    print("saved log:", LOG_PATH)


# final:
# 전체 train set으로 학습한 마지막 checkpoint 기준 test accuracy 기록
else:
    test_acc = evaluate(test_loader)

    torch.save(model.state_dict(), CKPT_PATH)

    print(f"final test acc: {test_acc:.2f}%")

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            RUN_ID,
            MODE,
            "final_test",
            BATCH_SIZE,
            EPOCHS,
            LR,
            OPTIMIZER_NAME,
            WEIGHT_DECAY,
            SEED,
            "",
            "",
            "",
            test_acc,
        ])

    print("saved final checkpoint:", CKPT_PATH)
    print("saved log:", LOG_PATH)