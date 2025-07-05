You are an expert in language model tuning. You are going to adjust the learning rate of finetuning GPT-2 model using WikiText-2 train data. Your goal is to minimize the final validation loss. 

# Log History

{{log_history}}

# Instruction

- Based on the log history, you need to decide how to change the learning rate. 

- You MUST output choose between following three actions: 
    1. "Double", double the learning rate 
    2. "Half", reduce the learning rate to 50% of the learning rate. 
    3. "Same", make no change.

- Respond in json format with explanation within 100 words following: 

{
    "explanation": <explanation in 100 words>
    "action": <"Doube", "Half" or "Same">
}

# Response
