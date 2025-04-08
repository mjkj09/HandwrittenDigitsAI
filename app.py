import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas
from CNN import CNN

# Configuration constants
CANVAS_SIZE = 280  # Size of the drawing canvas
MODEL_PATH = "models/mnist_cnn_2ep_93.94acc_20250407_225823.pth"
CLASSES = 10  # Number of digit classes (0-9)
COLUMNS_PER_ROW = 5  # Number of probability bars per row

# Visual constants
BACKGROUND_COLOR = "#000000"  # Black canvas background
STROKE_COLOR = "#FFFFFF"  # White drawing color
STROKE_WIDTH = 18  # Drawing stroke width


@st.cache_resource
def load_model() -> CNN:
    """Load trained PyTorch model from disk (CPU-only)."""
    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def process_image(image_data: np.ndarray) -> torch.Tensor | None:
    """Convert canvas RGBA image to grayscale 28x28 tensor with normalization."""
    if image_data is None:
        return None

    try:
        # Check if canvas is empty (all pixels are background)
        if not np.any(image_data[:, :, :-1]):
            return None

        # Convert numpy array to PIL Image and preprocess
        img = Image.fromarray(image_data.astype("uint8"), "RGBA")
        img = img.convert("L").resize((28, 28))

        # Apply same transforms as during training
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        return transform(img).unsqueeze(0)
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return None


def display_header() -> None:
    """Display application header with logo."""
    # PWA manifest and theme setup
    st.markdown(
        """
        <link rel="manifest" href="/manifest.json">
        <meta name="theme-color" content="#0E1117">
    """,
        unsafe_allow_html=True,
    )

    # Logo display
    logo = Image.open("images/genesys_logo.png")
    col1, _ = st.columns([1, 5])
    with col1:
        st.image(logo, width=300)


def display_prediction(probs: torch.Tensor, pred: int) -> None:
    """Display prediction results with probability distribution."""
    st.subheader("Rozkład prawdopodobieństwa")
    st.metric(label="Cyfra", value=pred)

    # Create 2 rows of 5 columns each
    rows = [st.columns(COLUMNS_PER_ROW) for _ in range(2)]
    widgets = {}

    # Initialize progress bars for each digit
    for i in range(CLASSES):
        row_idx = i // COLUMNS_PER_ROW
        col_idx = i % COLUMNS_PER_ROW

        with rows[row_idx][col_idx]:
            widgets[i] = st.progress(0, text=f"{i}: 0.0%")

    # Update progress bars with actual probabilities
    for i, p in enumerate(probs.squeeze().numpy()):
        widgets[i].progress(float(p), text=f"{i}: {p * 100:.1f}%")


def main() -> None:
    """Main application entry point."""
    st.set_page_config(layout="wide", page_title="Undertrained")
    display_header()

    # Initialize session state
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    # Load trained model
    model = load_model()
    st.title(
        "Rozpoznawanie cyfr - model na jednej warstwie konwolucyjnej o gorszych parametrach"
    )

    # Create main layout columns
    col1, col2 = st.columns(2)

    with col1:
        # Drawing canvas setup
        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=STROKE_WIDTH,
            stroke_color=STROKE_COLOR,
            background_color=BACKGROUND_COLOR,
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            display_toolbar=False,
            update_streamlit=True,
            key=f"canvas_{st.session_state.canvas_key}",
        )

        # Canvas reset button
        if st.button("Wyczyść obrazek"):
            st.session_state.canvas_key += 1
            st.rerun()

    # Process drawing and display results
    if canvas.image_data is not None:
        input_tensor = process_image(canvas.image_data)

        if input_tensor is not None:
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs).item()

            with col2:
                display_prediction(probs, pred)

        else:
            # Empty canvas - display zero probabilities
            with col2:
                zero_probs = torch.zeros((1, CLASSES))
                display_prediction(zero_probs, pred="-")

if __name__ == "__main__":
    main()
