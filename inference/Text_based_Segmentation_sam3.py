import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import Sam3Processor, Sam3Model  # Using specific SAM3 classes

# path setup : files
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "data" / "raw" / "image_007.jpg"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found at: {IMAGE_PATH}")

# loading sam3 model & processor
print(f"Initializing SAM 3 on {DEVICE}...")
# We use the processor to handle the text-to-tensor conversion
processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained(
    "facebook/sam3",
    trust_remote_code=True
).to(DEVICE)

# defining targets
# sam3 can understand concept prompts or noun phrases accprding to a paper: sam2 to sam3 gaps in SAM-family
target_objects = [
    "glasses",
    "cycle",
    "cyclist",
    "little girl",
    "boy in red shirt"
]

# inference
image_pil = Image.open(IMAGE_PATH).convert("RGB")
width, height = image_pil.size

print(f"Running inference on image ({width}x{height})...")

for target in target_objects:
    print(f" > Segmenting: '{target}'...")

    # 1. prepare inputs (Image + Text Prompt)
    inputs = processor(
        images=image_pil,
        text=target,
        return_tensors="pt"
    ).to(DEVICE)

    # 2. forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # 3. post-process (convert raw logits to boolean masks)
    # This aligns the masks back to the original image size
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,  # confidence threshold
        mask_threshold=0.5,  # pixel probability threshold
        target_sizes=[(height, width)]  # must be list of tuples
    )[0]

    # saving results
    found_masks = results["masks"]  # shape: (N, H, W)
    scores = results["scores"]  # shape: (N,)

    if len(found_masks) == 0:
        print(f"   [!] No objects found for '{target}'")
        continue

    # combine all instances of this object into one file (optional)
    # or save them individually. savning the best one for the object.
    # save the highest scoring mask for this text prompt
    best_idx = scores.argmax().item()
    best_mask = found_masks[best_idx].cpu().numpy()  # Boolean array
    best_score = scores[best_idx].item()

    clean_name = target.replace(" ", "_")
    filename = OUTPUT_DIR / f"mask_{clean_name}.npy"

    np.save(filename, best_mask)

    # optional: save a visual debug image too
    # mask_img = Image.fromarray((best_mask * 255).astype(np.uint8))
    # mask_img.save(OUTPUT_DIR / f"view_{clean_name}.png")

    print(f"   - Saved mask to {filename.name} (Confidence score is: {best_score:.3f})")

print(f"\nProcess complete. Check: {OUTPUT_DIR}")