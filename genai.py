# STEP 1: Installations
!pip install transformers datasets pandas
# STEP 2: Imports
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["WANDB_DISABLED"] = "true"

# STEP 3 :Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Reading the CSV from your Google Drive main directory
movies_df = pd.read_csv('/content/drive/My Drive/movies_dataset.csv')
print(f"Successfully loaded dataset with {len(movies_df)} rows")

# STEP 4 : Display the first few rows to check the data
print("\nPreview of the dataset:")
print(movies_df.head())

# Print column names to help with preprocessing
print("\nAvailable columns:")
print(movies_df.columns.tolist())

# STEP 5 : Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(movies_df, preserve_index=False)


def preprocess(example):
    title = example.get("title", "")
    overview = example.get("overview", "")

    return {
        "text": f"Movie: {title}\nOverview: {overview}"
    }

processed_dataset = dataset.map(preprocess)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# STEP 6: Tokenize dataset
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = processed_dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# STEP 7: Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# STEP 8: Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-movies-finetuned",
    run_name="gpt2-movies-run",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    save_steps=200,
    logging_steps=50,
    save_total_limit=1,
    fp16=True,
    report_to=[] if os.environ.get("WANDB_DISABLED", "false").lower() == "true" else ["wandb"]
)

# STEP 9: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

try:
    # STEP 10: Train
    trainer.train()

    # STEP 11: Save model and tokenizer
    trainer.save_model("./gpt2-movies-finetuned")
    tokenizer.save_pretrained("./gpt2-movies-finetuned")
    print("Training completed successfully and model saved!")
except Exception as e:
    print(f"An error occurred during training: {e}")
    print("You might want to try with a smaller dataset or adjust parameters.")
