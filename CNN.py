import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Simple Convolutional Neural Network for MNIST digit recognition.
    Includes two convolutional layers followed by BatchNorm, ReLU,
    MaxPooling, and Dropout for regularization. Finally, two fully
    connected layers for classification into 10 digits.
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.fc1 = nn.Linear(32 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 10)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights to improve convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the network.
        :param x: input tensor (batch_size x 1 x 28 x 28 for MNIST).
        :return: logits for 10 classes (batch_size x 10).
        """
        x = self.conv1(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)
