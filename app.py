import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

CANVAS_SIZE = 280
MODEL_PATH = "mnist_cnn_worst.pth"


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def process_image(image_data):
    if image_data is None:
        return None

    img = Image.fromarray(image_data.astype("uint8"), "RGBA")
    img = img.convert("L").resize((28, 28))

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    return transform(img).unsqueeze(0)


def main():
    st.title("Real-Time Digit Recognition")

    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        update_streamlit=True,
        key="canvas",
    )

    model = load_model()

    if canvas.image_data is not None:
        input_tensor = process_image(canvas.image_data)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs).item()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Rysunek")
            st.image(Image.fromarray(canvas.image_data[:, :, 3]), width=150)

        with col2:
            st.subheader("Predykcja")
            st.metric(label="Cyfra", value=pred)
            st.write("Prawdopodobieństwa:")
            for i, p in enumerate(probs.squeeze().numpy()):
                st.progress(float(p), text=f"{i}: {p*100:.1f}%")


if __name__ == "__main__":
    main()
