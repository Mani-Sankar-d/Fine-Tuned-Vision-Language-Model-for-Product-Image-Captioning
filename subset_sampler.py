import pandas as pd, os, shutil

# Load the dataset metadata
df = pd.read_csv("data/fashion-dataset/styles.csv", on_bad_lines='skip')

# Sample 10,000 products for faster training
subset = df.sample(10000, random_state=42)

# Create output folder for sampled images
os.makedirs("data/fashion-dataset/subset_images", exist_ok=True)

# Copy sampled images into subset folder
for img_id in subset["id"]:
    src = f"data/fashion-dataset/images/{img_id}.jpg"
    dst = f"data/fashion-dataset/subset_images/{img_id}.jpg"
    if os.path.exists(src):
        shutil.copy(src, dst)

# Helper function to generate short descriptive captions
def make_caption(r):
    parts = []
    if pd.notna(r["gender"]): parts.append(r["gender"])
    if pd.notna(r["baseColour"]): parts.append(r["baseColour"])
    if pd.notna(r["articleType"]): parts.append(r["articleType"])
    if pd.notna(r["usage"]): parts.append(f"for {r['usage']} wear")
    return " ".join(parts)

# Apply caption function and add image paths
subset["caption"] = subset.apply(make_caption, axis=1)
subset["image_path"] = subset["id"].apply(lambda x: f"subset_images/{x}.jpg")

# Save final dataset
subset[["image_path", "caption"]].to_csv("data/fashion-dataset/product_captions_filtered.csv", index=False)

print("Created product_captions_filtered.csv with 10K samples.")
