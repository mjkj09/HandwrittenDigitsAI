import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms

from CNN import CNN  # Import the same CNN definition used for training.

CANVAS_SIZE = 280
MODEL_PATH = "models/mnist_cnn_34ep_99.63acc_20250331_022317.pth"


@st.cache_resource
def load_model():
    """
    Load the trained PyTorch model from disk (CPU-only).
    """
    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def process_image(image_data):
    """
    Convert the drawn RGBA image to grayscale 28x28 tensor,
    then apply the same normalization used in training.
    """
    if image_data is None:
        return None

    # Convert to PIL Image
    img = Image.fromarray(image_data.astype("uint8"), "RGBA")

    # Convert to grayscale and resize to 28x28 (MNIST size)
    img = img.convert("L").resize((28, 28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Add batch dimension
    return transform(img).unsqueeze(0)


def main():
    st.title("Real-Time Digit Recognition")

    # Create a drawing canvas in Streamlit
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

    # Load the model (cached by Streamlit)
    model = load_model()

    # Process the image from the canvas
    if canvas.image_data is not None:
        input_tensor = process_image(canvas.image_data)

        if input_tensor is not None:
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs).item()

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Drawing")
                # Show only the alpha channel as a quick preview
                st.image(Image.fromarray(canvas.image_data[:, :, 3]), width=150)

            with col2:
                st.subheader("Prediction")
                st.metric(label="Digit", value=pred)
                st.write("Probabilities:")
                for i, p in enumerate(probs.squeeze().numpy()):
                    # Show progress bar for each digit probability
                    st.progress(float(p), text=f"{i}: {p * 100:.1f}%")


if __name__ == "__main__":
    main()
