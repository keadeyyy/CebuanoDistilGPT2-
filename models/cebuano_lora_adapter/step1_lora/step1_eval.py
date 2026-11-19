import sys, math, torch
sys.stdout.reconfigure(encoding="utf-8")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token

raw = load_dataset("text", data_files="cebuano_corpus.txt", encoding="utf-8")["train"]
raw = raw.filter(lambda x: len(x["text"].strip()) > 0)
tokd = raw.map(lambda b: tok(b["text"]), batched=True, remove_columns=["text"])

def group(ex, bs=128):
    j = sum(ex["input_ids"], [])
    n = (len(j) // bs) * bs
    return {"input_ids": [j[i:i + bs] for i in range(0, n, bs)]}

lm = tokd.map(group, batched=True, remove_columns=tokd.column_names)
test = lm.train_test_split(test_size=0.05, seed=42)["test"]
print("Held-out blocks:", len(test))

def perplexity(model, max_blocks=200):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, min(len(test), max_blocks), 8):
            ids = torch.tensor(test[i:i + 8]["input_ids"])
            losses.append(model(input_ids=ids, labels=ids).loss.item())
    return math.exp(sum(losses) / len(losses))

base = AutoModelForCausalLM.from_pretrained("distilgpt2")
tuned = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained("distilgpt2"), "cebuano_lora_adapter"
)

ppl_base = perplexity(base)
ppl_tuned = perplexity(tuned)
print("Baseline perplexity  :", round(ppl_base, 2))
print("Fine-tuned perplexity:", round(ppl_tuned, 2))

prompts = [
    "Kumusta ka?",
    "Ang Cebu usa ka",
    "Sa merkado, ang mga tawo",
]

with open("results_step1.txt", "w", encoding="utf-8") as f:
    f.write(f"Held-out blocks: {len(test)} (scored first 200)\n")
    f.write(f"Baseline perplexity  : {ppl_base:.2f}\n")
    f.write(f"Fine-tuned perplexity: {ppl_tuned:.2f}\n\n===== GENERATIONS =====\n")
    for name, m in [("BASE", base), ("TUNED", tuned)]:
        m.eval()
        for p in prompts:
            out = m.generate(
                **tok(p, return_tensors="pt"),
                max_new_tokens=50, do_sample=False,
                no_repeat_ngram_size=3, pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(out[0], skip_special_tokens=True)
            line = f"[{name}] {text}\n"
            print(line)
            f.write(line + "\n")
