import torch
from torchvision import datasets, transforms
import CNN


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=30,
                    translate=(0.5, 0.5),
                    scale=(0.25, 1.0),
                    shear=(-30, 30, -30, 30),
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=100,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=100,
    shuffle=True,
)


def train(model, device, train_loader, epochs, optimizer, loss_func, scheduler):
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)[0]
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch} - Loss: {loss.item():.6f}")


def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)[0]
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f"Accuracy: {correct / total:.6f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN.CNN().to(device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
epochs = 30
train(model, device, train_loader, epochs, optimizer, loss_func, scheduler)
test(model, device, test_loader)
torch.save(model.state_dict(), "mnist_cnn1.pth")
