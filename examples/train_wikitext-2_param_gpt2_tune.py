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


def main():
    model_name = "openai-community/gpt2"
    data_name = "wikitext"
    data_part = "wikitext-2-raw-v1"

    # Grid search parameters
    learning_rates = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4]

    # Load and prepare data once
    train_data = load_dataset(data_name, data_part, split="train")
    eval_data = load_dataset(data_name, data_part, split="validation")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )
    # Grid search loop
    for lr in learning_rates:
        # Initialize new wandb run for each learning rate
        run = wandb.init(
            project="interactive-trainer-wikitext-emnlp_gpt2_tune",
            name=f"lr_{lr}",
            config={"learning_rate": lr},
            reinit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name)

        args = TrainingArguments(
            output_dir=f"./wikitext2_lr_{lr}",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=1,
            num_train_epochs=5,
            learning_rate=lr,
            logging_steps=10,
            save_steps=1000,
            eval_steps=1000,
            eval_strategy="steps",
            fp16=True,
            report_to="wandb",
            eval_on_start=True,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=tokenizer,
            data_collator=collator,
        )

        print(f"Training with learning rate: {lr}")
        trainer.train()

        # Finish the current wandb run
        wandb.finish()


if __name__ == "__main__":
    main()
