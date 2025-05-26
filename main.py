import os
import time
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from trainer.interactive_callback import InteractiveTrainerCallback
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)


def main():
    os.environ["SERVER_HOST"] = "localhost"
    os.environ["SERVER_PORT"] = "9876"

    raw_datasets = load_dataset("imdb")  # train 25 000 / test 25 000
    label2id = {"neg": 0, "pos": 1}
    id2label = {v: k for k, v in label2id.items()}

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=256
        )  # 256 ≃ GPU-friendly, adjust if you like

    tokenized = raw_datasets.map(tokenize_fn, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01,
    )

    cb = InteractiveTrainerCallback(
        os.getenv("SERVER_HOST", "localhost"),
        os.getenv("SERVER_PORT", "9876"),
        model=model,
        optimizer=optimizer,
    )

    args = TrainingArguments(
        output_dir="bert-imdb-sentiment",
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        save_strategy="epoch",
        logging_steps=100,
        report_to="wandb",  # set to "tensorboard" or "wandb" if you prefer
    )

    trainer = Trainer(
        model=model,
        optimizers=(optimizer, None),  # No scheduler for simplicity
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],  # IMDb has a ready-made test split
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[cb],
    )

    trainer.train()


if __name__ == "__main__":
    main()
