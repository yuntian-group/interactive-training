import wandb
import argparse
from datasets import load_dataset
from transformers import (
    Trainer,
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
)

from interactive_training import make_interactive


def main(args):
    model_name = "openai-community/gpt2"
    data_name = "wikitext"
    data_part = "wikitext-2-raw-v1"
    wandb.init(project="interactive-trainer-wikitext")
    config = AutoConfig.from_pretrained(model_name)
    model = GPT2LMHeadModel(config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    args = TrainingArguments(
        output_dir="./wikitext2",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=1000,
        eval_steps=1000,
        eval_strategy="steps",
        fp16=True,
        report_to="wandb",
        eval_on_start=False,
    )

    train_data = load_dataset(data_name, data_part, split="train")
    eval_data = load_dataset(data_name, data_part, split="validation")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=1024, padding="longest"
        )

    train_data = train_data.filter(lambda x: len(x["text"]) > 0).map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    eval_data = eval_data.filter(lambda x: len(x["text"]) > 0).map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    InteractiveTrainer = make_interactive(Trainer)

    trainer = InteractiveTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 on WikiText-2")
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    args = parser.parse_args()
    main(args)
