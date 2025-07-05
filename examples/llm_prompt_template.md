You are an expert in language model tuning. You are going to adjust the learning rate of fine-tuning a GPT-2 model using the WikiText-2 train data with a constant learning rate scheduler. Your goal is to minimize the final validation loss.

# Log History

current step: {{current_step}}

current learning rate: 

{{current_lr}}

learning rate history: 

{{lr_history}}

train loss history: 

{{train_loss_history}}

validation loss history: 

{{valid_loss_history}}


# Instruction

- Based on the log history, decide how to change the learning rate. The log history includes metrics like validation loss, training loss, and epoch count.

- You MUST output choose between following three actions: 
    1. "Double", double the learning rate 
    2. "Half", reduce the learning rate to 50% of the learning rate. 
    3. "Same", make no change.

- Respond in JSON format with an explanation within 100 words:

{
    "explanation": "<Explanation in 100 words>",
    "action": "<'Double', 'Half', or 'Same'>"
}

# Response
