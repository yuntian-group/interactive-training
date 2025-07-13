import random
import argparse
from typing import List
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM
from interactive_training import make_interactive, make_interactive_dataset


import transformers

transformers.logging.set_verbosity_info()


class ExampleDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_source: List[str],
        max_length: int = 2048,
        num_samples: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

        self.dataset_sources = data_source
        self.sampling_probabilities = [0] * len(data_source)
        self.sampling_probabilities[0] = 1.0
        self.test_params = 0.11
        self.num_samples = num_samples
        self.max_length = max_length

        self.dataset_dict = {}

        for name in self.dataset_sources:
            if name.find("gsm8k") != -1:
                dataset = load_dataset(name, "main", split="train")
            else:
                dataset = load_dataset(name, split="train")
            self.dataset_dict[name] = self.process_dataset(name, dataset)

    def process_dataset(self, name, dataset):
        if name == "openai/gsm8k":
            # gsm8k has 'question' and 'answer' columns
            dataset = dataset.map(
                lambda x: {
                    "text": self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": x["question"]},
                            {"role": "assistant", "content": x["answer"]},
                        ],
                        tokenize=False,
                        enable_thinking=False,
                    )
                }
            )
        elif name == "sahil2801/CodeAlpaca-20k":
            # CodeAlpaca has 'instruction', 'input', 'output'
            def format_instruction(x):
                if x["input"]:
                    return f"""{x['instruction']}\n\n{x['input']}"""
                return x["instruction"]

            dataset = dataset.map(
                lambda x: {
                    "text": self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": format_instruction(x)},
                            {"role": "assistant", "content": x["output"]},
                        ],
                        tokenize=False,
                        enable_thinking=False,
                    )
                }
            )
        else:
            raise ValueError(f"Processing for dataset {name} is not implemented.")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=dataset.column_names
        )
        return tokenized_dataset

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        print(self.test_params)
        r = random.random()
        cum_prob = 0.0
        for ds_idx, prob in enumerate(self.sampling_probabilities):
            cum_prob += prob
            if r < cum_prob:
                break

        chosen_ds = self.dataset_dict[self.dataset_sources[ds_idx]]

        sample_idx = random.randint(0, len(chosen_ds) - 1)
        return chosen_ds[sample_idx]


def main(args):
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    InteractiveTrainer = make_interactive(Trainer)
    InteractiveExampleDataset = make_interactive_dataset(ExampleDataset)

    dataset = InteractiveExampleDataset(
        tokenizer=tokenizer,
        data_source=["openai/gsm8k"],  # "sahil2801/CodeAlpaca-20k"],
        num_samples=2048,
        interactive_parameter_names=["sampling_probabilities", "test_params"],
    )

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template="<|im_start|>assistant\n",
        instruction_template="<|im_start|>user\n",
        pad_to_multiple_of=8,
    )

    args = TrainingArguments(
        output_dir="./example_dataset_mix_reload",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=1000,
        fp16=True,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name)

    trainer = InteractiveTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with mixed datasets")
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    args = parser.parse_args()
    main(args)
