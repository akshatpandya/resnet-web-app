"""ImageNet Image Classification Web App - Streamlit."""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from datetime import datetime, timezone
import csv
import io

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


def get_model_input_preview_transform():
    """Transform to preview the actual model input image (pre-normalization)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
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


class SquareOverlayTransformer(VideoTransformerBase):
    """
    Outputs a square viewfinder (W x W), centered, with a square frame overlay
    whose side equals the window width.
    """

    def __init__(self) -> None:
        self.last_bgr_frame: np.ndarray | None = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # 1) Center-crop the largest square that fits, then
        # 2) Resize it to (w x w) so the displayed viewfinder is square
        #    and the overlay square side equals the window width.
        crop_side = min(w, h)
        x0 = (w - crop_side) // 2
        y0 = (h - crop_side) // 2
        crop = img[y0 : y0 + crop_side, x0 : x0 + crop_side]
        square = cv2.resize(crop, (w, w), interpolation=cv2.INTER_AREA)

        # Draw the square frame as the full border of the square viewfinder.
        cv2.rectangle(square, (0, 0), (w - 1, w - 1), (255, 255, 255), 3)

        self.last_bgr_frame = square
        return square


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _ensure_session_state():
    if "inference_history" not in st.session_state:
        st.session_state["inference_history"] = []


def _append_inference_history(*, source: str, model_name: str, top5: list[tuple[str, float]]):
    st.session_state["inference_history"].append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "model": model_name,
            "top5": top5,
        }
    )


def _history_to_csv_bytes(history: list[dict]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "timestamp_utc",
            "source",
            "model",
            "rank1_class",
            "rank1_confidence",
            "rank2_class",
            "rank2_confidence",
            "rank3_class",
            "rank3_confidence",
            "rank4_class",
            "rank4_confidence",
            "rank5_class",
            "rank5_confidence",
        ]
    )
    for entry in history:
        top5 = entry.get("top5") or []
        padded = (top5 + [("", "")] * 5)[:5]
        row = [entry.get("timestamp_utc", ""), entry.get("source", ""), entry.get("model", "")]
        for cls, conf in padded:
            row.append(cls)
            row.append("" if conf == "" else f"{float(conf):.8f}")
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def main():
    _ensure_session_state()

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
        st.caption("Square viewfinder with centered frame overlay.")
        webrtc_ctx = webrtc_streamer(
            key="camera",
            video_transformer_factory=SquareOverlayTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )
        capture_clicked = st.button("Capture frame", use_container_width=True)
        file_img = st.file_uploader(
            "Or upload an image",
            type=["jpg", "jpeg", "png", "webp"],
        )

        captured_image: Image.Image | None = None
        if capture_clicked:
            transformer = webrtc_ctx.video_transformer
            if transformer is None:
                st.warning("Camera is not ready yet. Please wait for the video to start.")
            else:
                frame_bgr = getattr(transformer, "last_bgr_frame", None)
                if frame_bgr is None:
                    st.warning("No camera frame available yet. Try again in a moment.")
                else:
                    captured_image = bgr_to_pil(frame_bgr).convert("RGB")

        uploaded_image: Image.Image | None = None
        if file_img is not None:
            uploaded_image = Image.open(file_img).convert("RGB")

        image = captured_image or uploaded_image
        source_label = (
            "camera_capture"
            if captured_image is not None
            else (getattr(file_img, "name", "uploaded_image") if uploaded_image is not None else "")
        )

    with col2:
        if image is None:
            st.info(
                "Grant camera permission or upload an image to see predictions. "
                f"{model_name} is trained on ImageNet (1000 classes)."
            )
        else:
            model_input_preview = get_model_input_preview_transform()(image)
            st.image(
                model_input_preview,
                caption="Model input (resize + center-crop to 224×224, pre-normalization)",
                use_container_width=True,
            )

            with st.spinner("Running inference..."):
                model = load_model(model_name)
                labels = load_imagenet_labels()
                predictions = run_inference(model, image, top_n, labels)
                top5_for_history = run_inference(model, image, 5, labels)
                _append_inference_history(
                    source=source_label or "unknown",
                    model_name=model_name,
                    top5=top5_for_history,
                )

            st.subheader("Top predictions")
            for i, (label, prob) in enumerate(predictions, 1):
                pct = prob * 100
                st.progress(prob, text=f"{i}. **{label}** — {pct:.1f}%")

            # Table view
            st.dataframe(
                [
                    {"Model": model_name, "Rank": i, "Class": label, "Confidence (%)": f"{prob * 100:.2f}"}
                    for i, (label, prob) in enumerate(predictions, 1)
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.divider()
            st.subheader("Session: saved top-5 results")
            history = st.session_state.get("inference_history", [])

            cols = st.columns([1, 1, 1])
            with cols[0]:
                st.metric("Images inferred (this session)", len(history))
            with cols[1]:
                st.download_button(
                    "Download CSV",
                    data=_history_to_csv_bytes(history),
                    file_name="inference_history_top5.csv",
                    mime="text/csv",
                    use_container_width=True,
                    disabled=len(history) == 0,
                )
            with cols[2]:
                if st.button("Clear session history", use_container_width=True, disabled=len(history) == 0):
                    st.session_state["inference_history"] = []
                    st.rerun()

            if history:
                st.dataframe(
                    [
                        {
                            "Timestamp (UTC)": h["timestamp_utc"],
                            "Source": h["source"],
                            "Model": h["model"],
                            "Top-1": f'{h["top5"][0][0]} ({h["top5"][0][1]*100:.2f}%)' if h.get("top5") else "",
                            "Top-2": f'{h["top5"][1][0]} ({h["top5"][1][1]*100:.2f}%)' if len(h.get("top5") or []) > 1 else "",
                            "Top-3": f'{h["top5"][2][0]} ({h["top5"][2][1]*100:.2f}%)' if len(h.get("top5") or []) > 2 else "",
                            "Top-4": f'{h["top5"][3][0]} ({h["top5"][3][1]*100:.2f}%)' if len(h.get("top5") or []) > 3 else "",
                            "Top-5": f'{h["top5"][4][0]} ({h["top5"][4][1]*100:.2f}%)' if len(h.get("top5") or []) > 4 else "",
                        }
                        for h in history
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
