The base BLIP model is trained on general image–text pairs.
We fine-tuned it on 10K curated fashion product images (from the Kaggle Fashion Product Images Dataset) to specialize it for apparel captioning.

        Model	          BLEU	METEOR
Zero-shot (Base BLIP)	  0.0	  0.13
Fine-tuned BLIP (Ours)	0.0	  0.89

🟢 Result: Over a 6.7× improvement in semantic accuracy (METEOR)
Fine-tuned model now understands terms like "men blue t-shirt for sports wear" instead of generic "a man wearing clothes."

fashion-image-captioning/
│
├── data/
│   └── fashion-dataset/
│       ├── images/                       # Original Kaggle images
│       ├── subset_images/                # 10K sampled images
│       ├── styles.csv                    # Original metadata
│       └── product_captions_filtered.csv # Final subset CSV (used for training)
│
├── datasets/
│   ├── train_dataset.py                  # Dataset class for training
│   └── test_dataset.py                   # Dataset class for evaluation
│
├── fine_tuned_blip_fashion/              # Saved fine-tuned model (auto-created)
│
├── finetune.py                           # Fine-tuning script             
├── requirements.txt                      # All dependencies
└── README.md                             # Project documentation


📦 Dataset Preparation

You’ll need the Fashion Product Images (Small) dataset from Kaggle:
👉 https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

🧾 Step 1. Download the dataset

Make sure you have the Kaggle CLI installed and configured:

pip install kaggle


Then run:

kaggle datasets download -d paramaggarwal/fashion-product-images-small
unzip fashion-product-images-small.zip -d data/fashion-dataset/


This creates:

data/fashion-dataset/
 ├── images/
 ├── styles.csv

✂️ Step 2. Create a 10K subset and generate captions

Run subset_sampler.py once — it will:

Randomly sample 10,000 rows from styles.csv

Copy the corresponding images into a new folder

Build a caption using gender, color, article type, and usage

Save everything as product_captions_filtered.csv


📁 Resulting directory

After running the script, folder should look like this:

data/fashion-dataset/
 ├── images/                 # full dataset
 ├── subset_images/          # 10K sampled images
 ├── styles.csv              # original metadata
 └── product_captions_filtered.csv  # ready for training

# Step3. Execute finetune.py 


