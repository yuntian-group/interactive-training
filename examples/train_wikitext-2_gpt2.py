import wandb
import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

from src import make_interactive


def main():
    model_name = "openai-community/gpt2"
    data_name = "wikitext"
    data_part = "wikitext-2-raw-v1"
    wandb.init(project="interactive-trainer-wikitext")
    model = AutoModelForCausalLM.from_pretrained(model_name)
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
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=1000,
        eval_steps=1000,
        eval_strategy="steps",
        fp16=True,
        report_to="wandb",
        eval_on_start=True,
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

    interactive_trainer_cls = make_interactive(Trainer)

    trainer = interactive_trainer_cls(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
