import os
import datasets
from src import make_interactive
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def main():
    os.environ["SERVER_HOST"] = "localhost"
    os.environ["SERVER_PORT"] = "9876"

    raw_datasets = datasets.load_dataset("imdb")
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
        learning_rate=2.1415e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=300,
        weight_decay=0.0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    trainer_cls = make_interactive(Trainer)

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        # eval_dataset=tokenized["test"],  # IMDb has a ready-made test split
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()


if __name__ == "__main__":
    main()
