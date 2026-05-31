import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import urllib.request

# 自动下载MNIST
def download_mnist(root_dir="data"):
    raw_dir = os.path.join(root_dir, "MNIST", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    for f in files:
        path = os.path.join(raw_dir, f)
        if os.path.exists(path):
            print(f"{f} 已存在")
            continue
        print(f"下载 {base_url+f}")
        urllib.request.urlretrieve(base_url + f, path)

download_mnist()

# 超参数
batch_size = 64
lr = 0.001
epochs = 10
device = torch.device("cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_set = datasets.MNIST("data", train=True, download=False, transform=transform)
test_set = datasets.MNIST("data", train=False, download=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# ===================== 升级后的CNN模型 =====================
class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        # 28->14->7->3  128 * 3 * 3 = 1152
        self.fc_layers = nn.Sequential(
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x

model = BetterCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练&测试
train_loss_list, train_acc_list = [], []
test_loss_list, test_acc_list = [], []

def train(ep):
    model.train()
    total_loss = 0
    correct = 0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = torch.argmax(out, dim=1)
        correct += (pred == label).sum().item()

    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / len(train_set) * 100
    train_loss_list.append(avg_loss)
    train_acc_list.append(avg_acc)
    print(f"Epoch {ep:2d} | Train Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")

def evaluate(ep):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            out = model(data)
            loss = criterion(out, label)
            total_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            correct += (pred == label).sum().item()

    avg_loss = total_loss / len(test_loader)
    avg_acc = correct / len(test_set) * 100
    test_loss_list.append(avg_loss)
    test_acc_list.append(avg_acc)
    print(f"Epoch {ep:2d} | Test  Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%\n")

if __name__ == "__main__":
    for e in range(1, epochs+1):
        train(e)
        evaluate(e)

    # 保存新模型
    torch.save(model.state_dict(), "models/better_cnn_mnist.pth")
    print("新模型已保存至 models/better_cnn_mnist.pth")

    # 绘图
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(test_loss_list, label="Test Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Loss Curve")

    plt.subplot(1,2,2)
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(test_acc_list, label="Test Acc")
    plt.legend()
    plt.grid(True)
    plt.title("Accuracy Curve")
    plt.savefig("models/new_loss_acc.png")
    plt.close()
