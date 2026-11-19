from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling)
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

lora_config = LoraConfig(
    r=16,                        
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    fan_in_fan_out=True,            
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  

raw = load_dataset("text", data_files="cebuano_corpus.txt")["train"]

def tokenize(batch):
    return tokenizer(batch["text"])

tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])

def group_texts(examples, block_size=128):
    joined = sum(examples["input_ids"], [])
    total = (len(joined) // block_size) * block_size
    blocks = [joined[i:i + block_size] for i in range(0, total, block_size)]
    return {"input_ids": blocks, "attention_mask": [[1] * block_size] * len(blocks)}

lm_data = tokenized.map(group_texts, batched=True, remove_columns=tokenized.column_names)

split = lm_data.train_test_split(test_size=0.05, seed=42)

args = TrainingArguments(
    output_dir="lora_out",
    per_device_train_batch_size=8,
    max_steps=500,                 
    learning_rate=2e-4,
    logging_steps=20,
    save_strategy="steps", save_steps=100, save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
model.save_pretrained("cebuano_lora_adapter")   

logs = [l for l in trainer.state.log_history if "loss" in l]
plt.plot([l["step"] for l in logs], [l["loss"] for l in logs])
plt.xlabel("Step"); plt.ylabel("Training loss"); plt.title("LoRA fine-tuning")
plt.savefig("loss_curve.png")

import math
print("Perplexity:", math.exp(trainer.evaluate()["eval_loss"]))

model.save_pretrained("cebuano_lora_adapter")   
