# Interactive Training: Feedback-Driven Neural Network Optimization

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/yuntian-group/interactive-training)
[![Demo](https://img.shields.io/badge/Demo-Live-green)](https://interactivetraining.ai)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Interactive Training is an open-source framework that enables real-time, feedback-driven intervention during neural network training. Unlike traditional static training approaches, Interactive Training allows human experts or automated AI agents to dynamically adjust optimizer parameters, training data, and model checkpoints while training is in progress.

## 🎮 Try the Interactive Demo

Play our interactive game at [**interactivetraining.ai**](https://interactivetraining.ai/) experience the power of dynamic optimization control.

## 🚀 Key Features

- **Real-time Interventions**: Dynamically adjust learning rates, optimizer parameters, and training configurations during training
- **Interactive Dashboard**: React-based frontend for visualizing training metrics and sending control commands
- **Checkpoint Management**: Save, load, and branch training trajectories with full history tracking
- **AI Agent Support**: Enable LLM-based agents to automatically optimize training parameters
- **Easy Integration**: Minimal code changes required - just wrap your existing Hugging Face Trainer
- **Branching Support**: Create and manage multiple training branches from any checkpoint
- **WebSocket Communication**: Real-time bidirectional communication between training process and dashboard

## 🏗️ Architecture

Interactive Training consists of three main components:

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Frontend Dashboard │◄──►│   Control Server    │◄──►│ Interactive Trainer │
│   (React/TypeScript)│    │    (FastAPI)        │    │ (HuggingFace Trainer│
│                     │    │                     │    │     + Callbacks)    │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

- **Control Server**: FastAPI-based server that mediates communication between frontend and trainer
- **Interactive Trainer**: Extended Hugging Face Trainer with real-time intervention capabilities
- **Frontend Dashboard**: React-based visualization and control interface

## 📦 Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend development)

### Python Package Installation

```bash
# Clone the repository
git clone https://github.com/yuntian-group/interactive-training.git
cd interactive-training

# Install the package
pip install -e .
```

### Frontend Setup (Optional)

If you want to use the interactive dashboard:

```bash
cd frontend/interactive_optimizer
npm install
npm run build
```

## 🔧 Quick Start

### Basic Usage

Transform your existing Hugging Face training script with just 3 lines of code:

```python
from transformers import Trainer
from interactive_training import make_interactive  # 1. Import helper

# 2. Wrap the standard Trainer class
InteractiveTrainer = make_interactive(Trainer)

# 3. Use InteractiveTrainer exactly as you would the original Trainer
trainer = InteractiveTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()  # Training is now fully interactive!
```

### Complete Example

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from interactive_training import make_interactive, Trainer

def main():
    # Load model and tokenizer
    model_name = "openai-community/gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=1024, 
            padding="longest"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=1000,
        eval_steps=1000,
        evaluation_strategy="steps",
    )
    
    # Create interactive trainer
    InteractiveTrainer = make_interactive(Trainer)
    trainer = InteractiveTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Start training - now interactive!
    trainer.train()

if __name__ == "__main__":
    main()
```

## 🖥️ Interactive Dashboard

Start your training script, then open the interactive dashboard:

```bash
# Your training script will automatically start the control server
# Open your browser and navigate to:
http://localhost:9876
```

The dashboard provides:
- **Real-time metrics visualization** (loss, learning rate, gradient norms)
- **Control panels** for optimizer, checkpoints, and model management
- **Command history** and status tracking
- **Branching visualization** for experiment management

## 🤖 AI Agent Integration

Enable LLM-based agents to automatically optimize your training:

```python
# Example: LLM agent for learning rate optimization
import openai
from interactive_training.agents import LLMAgent

agent = LLMAgent(
    model="gpt-4",
    api_key="your-api-key",
    optimization_target="validation_loss"
)

# Agent will monitor training and suggest optimizations
agent.monitor_training(trainer, intervention_frequency=10)
```

## 📚 API Reference

### Supported Commands

| Command | Description | Parameters |
|---------|-------------|------------|
| `update_optimizer` | Modify optimizer parameters | `{"lr": 1e-4, "weight_decay": 0.01}` |
| `save_checkpoint` | Save current training state | `{}` |
| `load_checkpoint` | Load previous checkpoint | `{"uuid": "checkpoint_id", "branch_name": "new_branch"}` |
| `pause_training` | Pause training execution | `{}` |
| `resume_training` | Resume paused training | `{}` |
| `stop_training` | Stop training entirely | `{}` |
| `do_evaluate` | Trigger evaluation | `{}` |

### REST API Endpoints

- `GET /api/get_info/` - Get current training state
- `GET /api/get_optimizer_info/` - Get optimizer parameters
- `GET /api/get_model_info/` - Get model information
- `GET /api/get_checkpoints/` - Get saved checkpoints
- `GET /api/get_logs/` - Get training logs
- `POST /api/command/` - Send intervention command
- `WebSocket /ws/message/` - Real-time event stream

## 📁 Repository Structure

```
interactive_training/
├── src/                              # Core Python package
│   ├── __init__.py                   # Main package interface
│   ├── interactive_training_mixin.py # Interactive training mixin class
│   ├── interactive_training_server.py # FastAPI control server
│   ├── callbacks.py                  # Training callbacks for interventions
│   └── constants.py                  # Command constants and types
├── examples/                         # Example scripts and templates
│   ├── train_wikitext-2_gpt2.py     # Basic training example
│   ├── llm_as_tuner.py              # LLM agent example
│   └── llm_prompt_template.md       # LLM agent prompt template
├── frontend/                         # React-based dashboard
│   └── interactive_optimizer/        # Frontend application
│       ├── src/                      # React source code
│       │   ├── components/           # UI components
│       │   ├── features/             # Redux state management
│       │   └── api/                  # API client
│       └── package.json              # Frontend dependencies
├── exp1_data/                        # Experimental data (human vs static)
├── exp2_data/                        # Experimental data (LLM vs static)
├── pyproject.toml                    # Python package configuration
└── README.md                         # This file
```

### Key Files

- **`src/interactive_training_mixin.py`**: Core mixin class that adds interactivity to Hugging Face Trainer
- **`src/interactive_training_server.py`**: FastAPI server handling command/event routing
- **`src/callbacks.py`**: Training callbacks for different intervention types
- **`examples/train_wikitext-2_gpt2.py`**: Complete example showing basic usage
- **`examples/llm_as_tuner.py`**: Example of LLM-based automated optimization

## 🤝 Contributing

We welcome contributions!


## 📖 Citation

If you use Interactive Training in your research, please cite:

```bibtex
@article{interactive_training_2024,
  title={Interactive Training: Feedback-Driven Neural Network Optimization},
  author={Wentao Zhang, Yang Young Lu, Yuntian Deng},
  url={https://github.com/yuntian-group/interactive-training},
  year={2025}
}
```
