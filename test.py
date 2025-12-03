import torch
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
from datasets.test_dataset import FashionDataset
from torch.utils.data import DataLoader
import evaluate

def collate_fn(batch):
    images, captions = zip(*batch)
    return list(images), list(captions)

# Load finetuned model
processor = BlipProcessor.from_pretrained("fine_tuned_blip_fashion")
model = BlipForConditionalGeneration.from_pretrained(
    "fine_tuned_blip_fashion",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# Load dataset
df = pd.read_csv('data/fashion-dataset/product_captions.csv')
df = df.iloc[8000:].reset_index(drop=True)
dataset = FashionDataset(df)
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

# Generate predictions
predictions = []
with torch.no_grad():
    model.eval()
    for images, _ in tqdm(loader):
        inputs = processor(images=images, return_tensors="pt").to("cuda", torch.float16)
        out = model.generate(**inputs)
        for o in out:
            caption = processor.decode(o, skip_special_tokens=True)
            predictions.append(caption)

df["predicted_caption"] = predictions

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

preds = df["predicted_caption"].astype(str).tolist()
references = df["caption"].apply(lambda x: [str(x)]).tolist()

bleu_score = bleu.compute(predictions=preds, references=references)
meteor_score = meteor.compute(predictions=preds, references=references)

print("BLEU:", bleu_score)
print("METEOR:",meteor_score)