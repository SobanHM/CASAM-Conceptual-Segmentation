import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import pipeline

# --- 1. SETTINGS & PATHS ---
DEVICE = 0 if torch.cuda.is_available() else -1  # 0 for cuda:0

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "data" / "raw" / "image_015.jpg"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. LOAD PIPELINE ---
print(f"🚀 Initializing SAM 3 Pipeline on {'cuda:0' if DEVICE == 0 else 'cpu'}...")

# The mask-generation pipeline handles all the 'forward_image' logic internally
generator = pipeline(
    "mask-generation",
    model="facebook/sam3",
    device=DEVICE,
    trust_remote_code=True
)

# --- 3. RUN INFERENCE ---
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"❌ Image not found at: {IMAGE_PATH}")

image_pil = Image.open(IMAGE_PATH).convert("RGB")
width, height = image_pil.size

print("🧠 Running neural network inference...")

# We provide a 'points' prompt to the generator
# Format: [x, y]
input_points = [[width // 2, height // 2]]

outputs = generator(
    image_pil,
    points=input_points,
    multimask_output=True
)

# --- 4. PROCESSING & SAVING ---
# outputs is a list of dictionaries containing 'segmentation' (bool mask)
masks = outputs["masks"]
scores = outputs["scores"]

print(f"✅ Generated {len(masks)} mask variations.")

for i, (mask, score) in enumerate(zip(masks, scores)):
    # mask is a boolean numpy array
    mask_filename = OUTPUT_DIR / f"sam3_mask_var_{i}.npy"
    np.save(mask_filename, mask)
    print(f"   - Saved variation {i} (Score: {score:.3f})")

print(f"\n[OK] Process Complete. Check: {OUTPUT_DIR}")


# import torch
# import numpy as np
# from PIL import Image
# import os
#
# from transformers import SamModel, SamProcessor
#
# # ---------------------------
# # Configuration
# # ---------------------------
# DEVICE = torch.device("cuda:0")
# IMAGE_PATH = "data/raw/image_015.jpg"
# OUTPUT_DIR = "data/processed/"
#
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# # ---------------------------
# # Load SAM3 model + processor
# # ---------------------------
# processor = SamProcessor.from_pretrained("facebook/sam3")
# model = SamModel.from_pretrained("facebook/sam3").to(DEVICE)
# model.eval()
#
# # ---------------------------
# # Load Image
# # ---------------------------
# image = Image.open(IMAGE_PATH).convert("RGB")
#
# # ---------------------------
# # Conditioning (temporary point)
# # ---------------------------
# width, height = image.size
# input_points = [[[width // 2, height // 2]]]
#
# # ---------------------------
# # Preprocess
# # ---------------------------
# inputs = processor(
#     image,
#     input_points=input_points,
#     return_tensors="pt"
# ).to(DEVICE)
#
# # ---------------------------
# # Inference
# # ---------------------------
# with torch.no_grad():
#     outputs = model(**inputs)
#
# # ---------------------------
# # Extract masks
# # ---------------------------
# masks = outputs.pred_masks.squeeze(1).cpu().numpy()
# scores = outputs.iou_scores.cpu().numpy()
#
# # ---------------------------
# # Save outputs
# # ---------------------------
# for i, mask in enumerate(masks):
#     np.save(f"{OUTPUT_DIR}/sam3_mask_{i}.npy", mask)
#
# np.save(f"{OUTPUT_DIR}/sam3_scores.npy", scores)
#
# print(f"[OK] SAM3 inference completed. {len(masks)} masks saved.")
#
#
# # import torch
# # print("import torch")
# # import numpy as np
# # print("import numpy")
# # from PIL import Image
# # print("import pillow")
# # import os
# # print("import os")
# #
# # from sam3.build_sam3 import build_sam3
# # from sam3.sam3_predictor import SAM3Predictor
# #
# # print("import sam3 predictor")
# #
# # # ---------------------------
# # # Configuration
# # # ---------------------------
# # DEVICE = torch.device("cuda:0")
# # IMAGE_PATH = "data/raw/sample.jpg"
# # OUTPUT_DIR = "data/processed/"
# #
# # os.makedirs(OUTPUT_DIR, exist_ok=True)
# #
# # # ---------------------------
# # # Load SAM3 Predictor
# # # ---------------------------
# # predictor = SamPredictor.from_pretrained(
# #     "facebook/sam3",
# #     device=DEVICE
# # )
# #
# # # ---------------------------
# # # Load Image
# # # ---------------------------
# # image = np.array(
# #     Image.open(IMAGE_PATH).convert("RGB")
# # )
# #
# # predictor.set_image(image)
# #
# # # ---------------------------
# # # Conditioning (temporary)
# # # One positive point in image center
# # # ---------------------------
# # h, w, _ = image.shape
# # point_coords = np.array([[w // 2, h // 2]])
# # point_labels = np.array([1])  # foreground
# #
# # # ---------------------------
# # # Inference
# # # ---------------------------
# # masks, scores, logits = predictor.predict(
# #     point_coords=point_coords,
# #     point_labels=point_labels,
# #     multimask_output=True
# # )
# #
# # # ---------------------------
# # # Save outputs
# # # ---------------------------
# # for i, mask in enumerate(masks):
# #     np.save(f"{OUTPUT_DIR}/sam3_mask_{i}.npy", mask)
# #
# # np.save(f"{OUTPUT_DIR}/sam3_scores.npy", scores)
# #
# # print(f"[OK] SAM3 inference completed. {len(masks)} masks saved.")
