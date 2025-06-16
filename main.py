import os
import datasets
from torch.utils.data import Dataset
from datasets import load_dataset
from trainer.interactive_training_wrapper import InteractiveTrainingWrapper
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


import pandas as pd

# Sample fake reviews and binary sentiment labels
data = {
    "text": [
        "Absolutely loved the movie, brilliant performance!",
        "Terrible plot and bad acting. Wouldn't recommend.",
        "An emotional rollercoaster, deeply moving.",
        "Waste of time. The story made no sense.",
        "A masterpiece. Every scene was captivating.",
        "Poorly directed and painfully slow.",
        "Heartwarming and inspiring.",
        "The worst movie I have ever watched.",
        "Great cinematography and a strong script.",
        "Unbearably boring and way too long.",
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # 1 = positive, 0 = negative
}

# Convert to pandas DataFrame
df = pd.DataFrame(data)

df.to_csv("fake_imdb_test.csv", index=False)


def main():
    os.environ["SERVER_HOST"] = "localhost"
    os.environ["SERVER_PORT"] = "9876"

    raw_datasets = datasets.load_dataset("csv", data_files="fake_imdb_test.csv")
    label2id = {"neg": 0, "pos": 1}
    id2label = {v: k for k, v in label2id.items()}

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=256
        )  # 256 ≃ GPU-friendly, adjust if you like

    tokenized = raw_datasets.map(tokenize_fn, batched=True, remove_columns=["text"])
    args = TrainingArguments(
        output_dir="./imdb_bert",
        eval_strategy="no",
        logging_strategy="steps",
        do_eval=False,
        logging_steps=100,
        learning_rate=1.1415e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=300,
        weight_decay=0.0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        # eval_dataset=tokenized["test"],  # IMDb has a ready-made test split
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    # print(trainer.lr_scheduler)

    # trainer.train()

    interactive_trainer = InteractiveTrainingWrapper(trainer)
    interactive_trainer.train()


if __name__ == "__main__":
    main()
