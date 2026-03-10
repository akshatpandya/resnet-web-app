"""ImageNet Image Classification Web App - Streamlit."""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Page config
st.set_page_config(
    page_title="ImageNet Classifier",
    page_icon="📷",
    layout="wide",
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
IMAGENET_CLASSES_PATH = BASE_DIR / "imagenet_classes.txt"


@st.cache_resource
def load_model(model_name: str):
    """Load a pretrained ImageNet model and cache it."""
    if model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == "ViT-B/16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    return model


@st.cache_data
def load_imagenet_labels():
    """Load ImageNet class labels from local file."""
    with open(IMAGENET_CLASSES_PATH) as f:
        return [line.strip() for line in f.readlines()]


def get_preprocess():
    """Return the standard ImageNet preprocessing transform."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def run_inference(model, image: Image.Image, top_n: int, labels: list[str]):
    """Run model inference and return top-N predictions."""
    preprocess = get_preprocess()
    img_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)

    top_probs, top_indices = torch.topk(probs[0], min(top_n, len(labels)))

    return [
        (labels[idx], float(prob))
        for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
    ]


def main():
    st.title("ImageNet Image Classifier")
    st.markdown(
        "Capture an image with your camera or upload a file. "
        "The app runs an ImageNet-pretrained model and shows the top predictions."
    )

    # Sidebar: model and Top N configuration
    st.sidebar.header("Settings")
    model_name = st.sidebar.selectbox(
        "Model architecture",
        options=["ResNet50", "ViT-B/16"],
        index=0,
    )
    top_n = st.sidebar.slider(
        "Number of top predictions",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    # Main area: Camera and file upload
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Capture or upload image")
        camera_img = st.camera_input("Take a photo")
        file_img = st.file_uploader(
            "Or upload an image",
            type=["jpg", "jpeg", "png", "webp"],
        )

        # Use camera image if available, else file upload
        image_source = camera_img if camera_img is not None else file_img

    with col2:
        if image_source is None:
            st.info(
                "Grant camera permission or upload an image to see predictions. "
                f"{model_name} is trained on ImageNet (1000 classes)."
            )
        else:
            image = Image.open(image_source).convert("RGB")
            st.image(image, caption="Input image", use_container_width=True)

            with st.spinner("Running inference..."):
                model = load_model(model_name)
                labels = load_imagenet_labels()
                predictions = run_inference(model, image, top_n, labels)

            st.subheader("Top predictions")
            for i, (label, prob) in enumerate(predictions, 1):
                pct = prob * 100
                st.progress(prob, text=f"{i}. **{label}** — {pct:.1f}%")

            # Table view
            st.dataframe(
                [
                    {"Rank": i, "Class": label, "Confidence (%)": f"{prob * 100:.2f}"}
                    for i, (label, prob) in enumerate(predictions, 1)
                ],
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
