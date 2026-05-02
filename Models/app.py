import streamlit as st
import numpy as np
import cv2
import pickle
from streamlit_drawable_canvas import st_canvas

# -------------------------
# Load model
# -------------------------
try:
    with open("Models/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# -------------------------
# Load preprocessing params
# -------------------------
try:
    mean = np.load("Models/mean.npy")
    std  = np.load("Models/std.npy")
except Exception as e:
    st.error(f"Mean/Std load error: {e}")
    st.stop()

# -------------------------
# Preprocessing: canvas → MNIST-compatible 28×28
# -------------------------
def preprocess_canvas(img_rgba):
    """
    Converts a 280×280 RGBA canvas into a 28×28 float array
    matching the MNIST pixel distribution.
    """

    # Step 1: RGBA → grayscale
    gray = cv2.cvtColor(img_rgba.astype("uint8"), cv2.COLOR_RGBA2GRAY)

    # Step 2: Crop to bounding box of drawn pixels
    coords = np.column_stack(np.where(gray > 30))
    if coords.size == 0:
        return None, "empty"

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = gray[y_min:y_max + 1, x_min:x_max + 1]

    # Step 3: Reject drawings that are too small (dots, accidental marks)
    # If the bounding box covers less than 2% of the canvas area, it's not a digit.
    h, w      = cropped.shape
    canvas_area = img_rgba.shape[0] * img_rgba.shape[1]
    if (h * w) < (canvas_area * 0.02):
        return None, "too_small"

    # Step 4: Clamp extreme aspect ratios — fixes "1", "7" being 1px wide after resize.
    # A drawn "1" can be 200px tall x 15px wide. Naive square-padding makes the
    # stroke only ~1px wide in 28x28 space. MNIST "1"s are 4-7px wide.
    # Fix: enforce minimum width = height / 3.
    if h > w * 3:
        min_w    = h // 3
        expanded = np.zeros((h, min_w), dtype=np.uint8)
        x_off_e  = (min_w - w) // 2
        expanded[:, x_off_e:x_off_e + w] = cropped
        cropped  = expanded
        h, w     = cropped.shape

    # Step 5: Pad into a square (center the digit)
    side   = max(h, w)
    square = np.zeros((side, side), dtype=np.uint8)
    y_off  = (side - h) // 2
    x_off  = (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cropped

    # Step 6: Add border padding — MNIST digits have ~4px empty border.
    # Without this, the model sees a digit flush with edges which it never
    # encountered during training.
    border = max(1, side // 7)
    padded = cv2.copyMakeBorder(
        square, border, border, border, border,
        borderType=cv2.BORDER_CONSTANT, value=0
    )

    # Step 7: Mild Gaussian blur to simulate MNIST's soft/scanned edges.
    # IMPORTANT: keep blur small — large blur + no contrast stretch was the
    # cause of the "8 bias" bug (blobs look like 8 to the model).
    # We use a fixed small kernel regardless of drawing size.
    padded = cv2.GaussianBlur(padded, (3, 3), sigmaX=1)

    # Step 8: Resize to 28×28
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    # NOTE: We intentionally do NOT contrast-stretch here.
    # Contrast stretch was amplifying dim blobs (dots, smudges) to full
    # brightness, making them look like 8s to the model. The raw pixel
    # intensity after resize is a meaningful signal — preserve it.

    # Step 9: Normalize to [0, 1]
    return resized / 255.0, "ok"


# -------------------------
# UI
# -------------------------
st.title("Handwritten Digit Classifier (Random Forest)")
st.write("Draw a digit (0–9) below using a thick stroke:")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,        # 20px on 280×280 → ~2px after resize to 28×28 (matches MNIST thickness)
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
)

# -------------------------
# Prediction
# -------------------------
if canvas_result.image_data is not None:
    try:
        processed, status = preprocess_canvas(canvas_result.image_data)

        if status == "empty":
            st.info("Canvas is empty — draw a digit first.")

        elif status == "too_small":
            # This catches dots and tiny accidental marks that would
            # previously get spurious votes for 8.
            st.warning("Drawing is too small — please draw a full digit.")

        else:
            # Show the 28×28 image the model actually receives
            st.image(processed, caption="What the model sees (28×28)", width=150, clamp=True)

            # Flatten to (1, 784)
            img_flat = processed.reshape(1, -1)

            # Standardize using training mean/std.
            # Clamp near-zero std values (background pixels) to avoid division blowup.
            safe_std = np.where(std < 1e-6, 1.0, std)
            img_std  = (img_flat - mean) / safe_std

            # Majority vote across all trees
            tree_preds = np.array([tree.predict(img_std) for tree in model.trees]).flatten()
            counts     = np.bincount(tree_preds)
            final_pred = counts.argmax()
            confidence = counts.max() / len(tree_preds)

            # Results
            st.subheader(f"Prediction: {final_pred}")
            st.write(f"Confidence: {confidence * 100:.1f}%")

            # Per-class vote breakdown
            st.write("Per-class votes:")
            vote_fracs = counts / len(tree_preds)
            for digit, frac in enumerate(vote_fracs):
                st.progress(float(frac), text=f"{digit}: {frac * 100:.1f}%")

    except Exception as e:
        st.error(f"Prediction error: {e}")