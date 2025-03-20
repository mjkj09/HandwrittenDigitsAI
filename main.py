import torch
import torch.nn as nn
from torchvision import datasets, transforms
from CNN import CNN
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


train_transform = transforms.Compose(
    [
        transforms.RandomAffine(
            degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=True, download=True, transform=train_transform),
    batch_size=64,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=False, transform=test_transform),
    batch_size=64,
    shuffle=True,
)


def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    return accuracy


def train(model, device, train_loader, optimizer, loss_func, scheduler):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step(total_loss / len(train_loader))
    return total_loss / len(train_loader)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)
loss_func = nn.CrossEntropyLoss()


epochs = 5
best_acc = 0

for epoch in range(1, epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, loss_func, scheduler)
    acc = test(model, device, test_loader)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "mnist_cnn_worst.pth")

    print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={acc:.2f}%")


y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.show()
