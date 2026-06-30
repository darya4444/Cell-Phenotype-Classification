import streamlit as st
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from model import EfficientNetClassifier

import cv2
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Untreated",
    "Paclitaxel",
    "Vorinostat"
]

CHANNEL_NAMES = [
    "Blue",
    "Green",
    "Red",
    "Yellow"
]

MEAN = torch.tensor(
    [0.0022, 0.0078, 0.0050, 0.0196],
    dtype=torch.float32
).view(4,1,1)

STD = torch.tensor(
    [0.0075, 0.0201, 0.0129, 0.0474],
    dtype=torch.float32
).view(4,1,1)


# ============================================================
# Load model
# ============================================================

@st.cache_resource
def load_model():

    model = EfficientNetClassifier(num_classes=3)

    model.load_state_dict(
        torch.load(
            "weights/best_model2.pth",
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()

# ============================================================
# Visualization
# ============================================================

def colorize_channel(channel, color):

    channel = channel.astype(np.float32)

    channel -= channel.min()

    channel /= (channel.max() + 1e-8)

    channel *= 255

    channel = channel.astype(np.uint8)

    rgb = np.zeros((*channel.shape, 3), dtype=np.uint8)

    if color == "red":
        rgb[:, :, 0] = channel

    elif color == "green":
        rgb[:, :, 1] = channel

    elif color == "blue":
        rgb[:, :, 2] = channel

    elif color == "yellow":
        rgb[:, :, 0] = channel
        rgb[:, :, 1] = channel

    return rgb

def normalize_channel(channel):
    channel = channel.astype(np.float32)
    channel -= channel.min()
    channel /= (channel.max() + 1e-8)
    return channel

def normalize_display(channel):
    channel = channel.astype(np.float32)

    if channel.max() > channel.min():
        channel = (channel - channel.min()) / (channel.max() - channel.min())

    return channel


def create_composite(blue, green, red, yellow):

    blue = normalize_display(blue)
    green = normalize_display(green)
    red = normalize_display(red)
    yellow = normalize_display(yellow)

    composite = np.zeros((blue.shape[0], blue.shape[1], 3), dtype=np.float32)

    # Blue
    composite[:, :, 2] += 0.9 * blue

    # Green
    composite[:, :, 1] += 0.9 * green

    # Red
    composite[:, :, 0] += 0.9 * red

    # Yellow contributes equally to red and green,
    # but with a reduced weight
    composite[:, :, 0] += 0.45 * yellow
    composite[:, :, 1] += 0.45 * yellow

    composite = np.clip(composite, 0, 1)

    return (255 * composite).astype(np.uint8)

# ============================================================
# Channel occlusion analysis
# ============================================================

def channel_occlusion(model, image, predicted_class):

    """
    Computes channel importance by occluding one channel at a time.
    """

    with torch.no_grad():

        original_logits = model(image)

        original_probs = torch.softmax(original_logits, dim=1)[0]

    original_conf = original_probs[predicted_class].item()

    importance = []

    for c in range(4):

        occluded = image.clone()

        # remove one channel
        occluded[:, c, :, :] = 0

        with torch.no_grad():

            logits = model(occluded)

            probs = torch.softmax(logits, dim=1)[0]

        new_conf = probs[predicted_class].item()

        importance.append(max(original_conf - new_conf, 0))

    importance = np.array(importance)

    if importance.sum() > 0:
        importance = importance / importance.sum()

    return importance

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Cell Phenotype Classification",
    layout="wide"
)

st.title("Cell Phenotype Classification")

st.write(
    """
Upload four immunofluorescence channels.

The model classifies the cell into one of three classes:

- Untreated
- Paclitaxel
- Vorinostat
"""
)

st.divider()

left, right = st.columns(2)

with left:

    blue_file = st.file_uploader(
        "Blue channel",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="blue"
    )

    green_file = st.file_uploader(
        "Green channel",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="green"
    )

with right:

    red_file = st.file_uploader(
        "Red channel",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="red"
    )

    yellow_file = st.file_uploader(
        "Yellow channel",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="yellow"
    )


if all([
    blue_file,
    green_file,
    red_file,
    yellow_file
]):

    # ==========================================================
    # Read images exactly as during training
    # ==========================================================

    blue = cv2.imdecode(
        np.frombuffer(blue_file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    green = cv2.imdecode(
        np.frombuffer(green_file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    red = cv2.imdecode(
        np.frombuffer(red_file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    yellow = cv2.imdecode(
        np.frombuffer(yellow_file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    blue_rgb = colorize_channel(blue, "blue")
    green_rgb = colorize_channel(green, "green")
    red_rgb = colorize_channel(red, "red")
    yellow_rgb = colorize_channel(yellow, "yellow")

    composite = create_composite(
        blue,
        green,
        red,
        yellow
    )
    # ==========================================================
    # Show uploaded images
    # ==========================================================

    st.divider()
    st.subheader("Uploaded Images")

    c0, c1, c2, c3, c4 = st.columns(5)

    c0.image(
        composite,
        caption="Composite",
        use_container_width=True
    )

    c1.image(
        blue_rgb,
        caption="Blue",
        use_container_width=True
    )

    c2.image(
        green_rgb,
        caption="Green",
        use_container_width=True
    )

    c3.image(
        red_rgb,
        caption="Red",
        use_container_width=True
    )

    c4.image(
        yellow_rgb,
        caption="Yellow",
        use_container_width=True
    )

    st.divider()

    run_prediction = st.button(
        "🔬 Run prediction",
        type="primary",
        use_container_width=True
)
    # ==========================================================
    # Same preprocessing as IFCellDataset
    # ==========================================================
    if run_prediction:
        blue_model = cv2.resize(blue, (224,224))
        green_model = cv2.resize(green, (224,224))
        red_model = cv2.resize(red, (224,224))
        yellow_model = cv2.resize(yellow, (224,224))

        image = np.stack(
            [
                blue_model,
                green_model,
                red_model,
                yellow_model
            ],
            axis=0
        )
        image = torch.from_numpy(image).float() / 255.0

        image = (image - MEAN) / STD

        image = image.unsqueeze(0).to(DEVICE)

        # ==========================================================
        # Prediction
        # ==========================================================
        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)[0]
            prediction = logits.argmax(dim=1).item()

        confidence = probs[prediction].item()

        importance = channel_occlusion(
            model,
            image,
            prediction
        )

        st.success(f"Predicted class: {CLASS_NAMES[prediction]}")
        st.metric("Confidence", f"{confidence*100:.2f}%")

        st.subheader("📊 Class probabilities")

        for i, class_name in enumerate(CLASS_NAMES):
            probability = probs[i].item()
            st.write(f"**{class_name}**")
            st.progress(probability)
            st.write(f"{probability*100:.2f}%")

    # ============================================================
    # Probability table
    # ============================================================

        st.subheader("Prediction Summary")

        summary = {
            "Class": CLASS_NAMES,
            "Probability (%)": [
                round(float(p) * 100, 2)
                for p in probs.cpu()
            ]
        }

        st.table(summary)

        st.divider()


        st.subheader("Channel importance (Occlusion Analysis)")

        colors = {
            "Blue": "🔵",
            "Green": "🟢",
            "Red": "🔴",
            "Yellow": "🟡"
        }

        # сортируем по важности
        sorted_channels = sorted(
            zip(CHANNEL_NAMES, importance),
            key=lambda x: x[1],
            reverse=True
        )

        for name, value in sorted_channels:

            st.write(
                f"{colors[name]} **{name}** — {value*100:.1f}%"
            )

            st.progress(float(value))
        st.divider()

    # ============================================================
    # Model information
    # ============================================================

    with st.expander("Model information"):

        st.markdown(
            """
**Architecture**

EfficientNet-B0

**Input**

- 4 fluorescence channels
- Image size: 224 × 224

**Classes**

- Untreated
- Paclitaxel
- Vorinostat

**Prediction**

The uploaded channels are combined into a single 4-channel tensor,
normalized using the training dataset statistics and passed through the
trained neural network.
"""
        )        