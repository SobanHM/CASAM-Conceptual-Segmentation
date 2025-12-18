import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# --- SETTINGS ---
BASE_DIR = Path(__file__).resolve().parent.parent
MASK_PATH = BASE_DIR / "data" / "processed" / "sam3_mask_var_0.npy"
IMAGE_PATH = BASE_DIR / "data" / "raw" / "image_015.jpg"


def identify():
    if not MASK_PATH.exists() or not IMAGE_PATH.exists():
        print("Missing files. Check your data folders.")
        return

    # 1. Load data
    mask = np.load(MASK_PATH)
    img = Image.open(IMAGE_PATH).convert("RGB")
    img_array = np.array(img)

    # 2. Create the Overlay
    # We create a red version of the image for the masked area
    overlay = img_array.copy()
    overlay[mask == 1] = [255, 0, 0]  # Turn the object area RED

    # 3. Plot for identification
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Identified Object")
    # Blend the red overlay with original for a "highlight" effect
    plt.imshow(img_array)
    plt.imshow(mask, cmap='Reds', alpha=0.4)  # 40% transparency
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    identify()