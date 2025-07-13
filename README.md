# Interactive Trainer
 
## Run Example

Update `init.sh` based on your information

```
source init.sh

python3 examples/train_example_gpt-2_wikitext.py
```


## Setup

```
pip install .
```


## Usage 


```

import Trainer from transformers 
import make_interactive import interactive_training 

# Make interactive
InteractiveTrainer = make_interactive(Trainer)

# Initialize training
trainer = InteractiveTrainer(...)

```






