import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Targeted FGSM
def fgsm_targeted(model, x, target, eps):
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True
    output = model(x_adv)
    loss = F.cross_entropy(output, target.to(device))
    model.zero_grad()
    loss.backward()
    grad_sign = x_adv.grad.data.sign()
    x_adv = x_adv - eps * grad_sign
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

# 훈련 함수
def train(model, train_loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# 성공률 매트릭스 계산 (출력만)
def targeted_success_matrix(model, test_loader, eps):
    model.eval()
    matrix = np.zeros((10, 10))

    for true_class in range(10):
        for target_class in range(10):
            if target_class == true_class:
                continue  # 자기 자신으로의 공격은 생략

            total = 0
            success = 0

            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                mask = label == true_class
                if mask.sum() == 0:
                    continue
                data = data[mask]
                target = torch.full_like(label[mask], target_class).to(device)
                data_adv = fgsm_targeted(model, data, target, eps)
                output_adv = model(data_adv)
                pred_adv = output_adv.argmax(dim=1)
                success += pred_adv.eq(target).sum().item()
                total += len(data)

            rate = 100 * success / total if total > 0 else 0
            matrix[true_class][target_class] = rate

    return matrix

# 메인 실행
if __name__ == "__main__":
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = SimpleCNN().to(device)
    train(model, train_loader, epochs=3)

    eps = 0.3
    print(f"\n[Targeted FGSM Success Rate Matrix] (eps = {eps})")
    matrix = targeted_success_matrix(model, test_loader, eps)

    # 출력
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            print(f"True: {i}, Target: {j} → Success Rate: {matrix[i][j]:.2f}%")
