import datetime
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms

from CNN import CNN

##############################################################################
#                            HYPERPARAMETERS                                 #
##############################################################################
EPOCHS = 5                # Number of training epochs
BATCH_SIZE = 64           # Batch size for DataLoader
LEARNING_RATE = 0.001     # Initial learning rate
WEIGHT_DECAY = 1e-5       # Weight decay (L2 regularization)

# Temporary file to store the best model during training
TEMP_MODEL_PATH = "best_model_temp.pth"

# Folders to save models and confusion matrices
MODEL_SAVE_DIR = "models"
CM_SAVE_DIR = "confusion_matrices"

# Ensure these folders exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(CM_SAVE_DIR, exist_ok=True)

##############################################################################
#                            DATA PREPARATION                                #
##############################################################################
def get_data_loaders(batch_size=64):
    """
    Create and return the training and testing DataLoader objects
    for the MNIST dataset with data augmentations (only in train).

    :param batch_size: how many samples per batch to load
    :return: (train_loader, test_loader)
    """
    # Data augmentation for the training set
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,         # random rotation +/- 10 degrees
            translate=(0.1, 0.1),   # random shift
            scale=(0.9, 1.1),       # random scale
            shear=10                # random shear
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
    ])

    # No augmentation for test set
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download (if not present) and create datasets
    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=test_transform
    )

    # Wrap datasets in DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True  # shuffle for better training
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

##############################################################################
#                            TRAINING FUNCTION                               #
##############################################################################
def train_one_epoch(model, train_loader, loss_func, optimizer):
    """
    Train the model for a single epoch on the given train_loader.

    :param model: the PyTorch model to be trained
    :param train_loader: DataLoader providing the training data
    :param loss_func: loss function (e.g., CrossEntropyLoss)
    :param optimizer: optimizer (e.g., Adam)
    :return: average loss over the entire epoch
    """
    model.train()               # set model to training mode
    total_loss = 0.0           # accumulate loss over batches

    for data, target in train_loader:
        optimizer.zero_grad()   # clear gradients from previous step

        # Forward pass
        output = model(data)
        loss = loss_func(output, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Return average loss for this epoch
    return total_loss / len(train_loader)

##############################################################################
#                             EVALUATION FUNCTION                            #
##############################################################################
def evaluate(model, test_loader):
    """
    Evaluate model performance on the test_loader.

    :param model: the PyTorch model to be tested
    :param test_loader: DataLoader providing the test data
    :return: (accuracy, y_true, y_pred)
    """
    model.eval()               # set model to evaluation (inference) mode
    correct = 0
    total = 0
    y_true, y_pred = [], []

    with torch.no_grad():      # no need to compute gradients in eval
        for data, target in test_loader:
            # Forward pass
            output = model(data)
            # Get the index of the class with the highest log-probability
            _, predicted = torch.max(output, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Save ground truth and predictions for confusion matrix
            y_true.extend(target.numpy())
            y_pred.extend(predicted.numpy())

    accuracy = correct / total
    return accuracy, y_true, y_pred

##############################################################################
#                                   MAIN                                     #
##############################################################################
def main():
    """
    Main training script that:
    1) Prepares data loaders
    2) Trains the CNN for EPOCHS
    3) Evaluates best model
    4) Saves final model with date/time and accuracy in the filename
    5) Generates and saves confusion matrix
    """
    # 1) Prepare data loaders
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)

    # 2) Initialize model, optimizer, and loss function
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_func = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_acc = 0.0      # track best accuracy found
    best_epoch = 0      # which epoch produced the best accuracy

    # 3) Training loop
    for epoch in range(1, EPOCHS + 1):
        avg_train_loss = train_one_epoch(model, train_loader, loss_func, optimizer)
        accuracy, _, _ = evaluate(model, test_loader)

        # Scheduler step based on train loss
        scheduler.step(avg_train_loss)

        # If current accuracy is better than all previous ones, save model
        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), TEMP_MODEL_PATH)

        print(f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")

    # After training is finished, load the best model (from TEMP_MODEL_PATH)
    model.load_state_dict(torch.load(TEMP_MODEL_PATH))

    # Evaluate once more to confirm final accuracy
    final_acc, y_true, y_pred = evaluate(model, test_loader)
    print(f"\nBest accuracy found: {best_acc * 100:.2f}% (epoch {best_epoch})")
    print(f"Final Accuracy (re-loaded best model): {final_acc * 100:.2f}%")

    # 4) Construct a filename containing date/time and best accuracy
    # For example:  "mnist_cnn_5ep_99.12acc_20250330_231045.pth"
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # You can choose to use best_acc or final_acc if they are the same
    final_model_name = f"mnist_cnn_{best_epoch}ep_{best_acc*100:.2f}acc_{now_str}.pth"
    final_model_path = os.path.join(MODEL_SAVE_DIR, final_model_name)

    # Save (or rename) the best model to the new filename
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final best model to: {final_model_path}")

    # 5) Generate confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")

    # We can also save the confusion matrix figure with the same naming scheme
    cm_filename = f"cm_{best_epoch}ep_{best_acc*100:.2f}acc_{now_str}.png"
    cm_path = os.path.join(CM_SAVE_DIR, cm_filename)
    plt.savefig(cm_path)
    plt.show()

    # Remove the temporary model file
    if os.path.exists(TEMP_MODEL_PATH):
        os.remove(TEMP_MODEL_PATH)


if __name__ == "__main__":
    main()
