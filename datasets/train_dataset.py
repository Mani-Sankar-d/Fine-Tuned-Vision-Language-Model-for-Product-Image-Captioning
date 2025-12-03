from torch.utils.data import Dataset
from PIL import Image
import os

class FashionCaptionDataset(Dataset):
    def __init__(self, df, processor, root ="data/fashion-dataset"):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.processor = processor
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        image_path = os.path.join(self.root, row['image_path'])
        caption = str(row['caption']).strip().lower()
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image,
                                text=caption,
                                padding = "max_length",
                                truncation = True,
                                max_length = 30,
                                return_tensors = 'pt'
                            )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs
