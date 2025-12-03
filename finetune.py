from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
import torch
from datasets.train_dataset import FashionCaptionDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float32,
    use_safetensors=True
).to(device)
for name, param in model.named_parameters():
    param.requires_grad = False
    if "crossattention" in name or "lm_head" in name:
        param.requires_grad = True

df = pd.read_csv("data/fashion-dataset/product_captions_filtered.csv")
train_df = df.iloc[:7000]
val_df = df.iloc[7000:8000]
train_dataset = FashionCaptionDataset(train_df,processor)
val_dataset = FashionCaptionDataset(val_df,processor)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)
num_epochs = 3
scaler = torch.cuda.amp.GradScaler()

trainable = [name for name, p in model.named_parameters() if p.requires_grad]
print(f"Trainable layers: {len(trainable)}")
print(trainable[:10])

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
        if torch.isnan(loss):
            print("NaN loss — skipping batch")
            continue

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = model(**batch, labels=batch["input_ids"])
                val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")

model.save_pretrained("fine_tuned_blip_fashion")
processor.save_pretrained("fine_tuned_blip_fashion")
