from torch.utils.data import Dataset
from PIL import Image
import os
class FashionDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        path = self.df.loc[idx,'image_path']

        caption = self.df.loc[idx,'caption']
        image = Image.open(os.path.join("D:/repos/Image_captioning/data/fashion-dataset/",path)).convert('RGB')
        caption = caption.strip().lower()
        return image, caption

