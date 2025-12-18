
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# --- 1. SETUP PATHS ---
# Finds the Project Root (Goes up one level from 'inference/')
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"

# --- 2. SELECT FILES ---
# Change this to whichever mask you want to see
mask_file = PROCESSED_DIR / "sam3_mask_var_0.npy"
# Optional: Load original image to see the comparison
original_image_path = RAW_DIR / "image_015.jpg"

if not mask_file.exists():
    print(f"Error: Could not find {mask_file}")
    exit()

# load data
mask = np.load(mask_file)

# visualize
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

# side A: the mask
ax[0].imshow(mask, cmap="gray")
ax[0].set_title("Binary Mask (NPY)")
ax[0].axis("off")

# side B: Overlay (on original image)
if original_image_path.exists():
    img = Image.open(original_image_path).convert("RGB")
    ax[1].imshow(img)
    # overlay the mask with 50% transparency (alpha)
    masked_overlay = np.ma.masked_where(mask == 0, mask)
    ax[1].imshow(masked_overlay, cmap="autumn", alpha=0.5)
    ax[1].set_title("Mask Overlaid on Original")
else:
    ax[1].text(0.5, 0.5, "Original Image\nNot Found", ha='center')

ax[1].axis("off")

# display saved mask
output_png = mask_file.with_suffix(".png")
plt.tight_layout()
plt.savefig(output_png)
print(f"✅ Comparison saved as: {output_png}")
plt.show()