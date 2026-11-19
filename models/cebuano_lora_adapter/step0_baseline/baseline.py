from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# 1) Tokenizer fragmentation: how many tokens does each word need?
cebuano_words = ["kumusta", "salamat", "maayong buntag", "balay", "gihigugma"]
english_words = ["hello", "thanks", "good morning", "house", "beloved"]

for ceb, eng in zip(cebuano_words, english_words):
    ceb_tokens = tokenizer.tokenize(ceb)
    eng_tokens = tokenizer.tokenize(eng)
    print(f"{ceb}: {len(ceb_tokens)} tokens {ceb_tokens}")
    print(f"{eng}: {len(eng_tokens)} tokens {eng_tokens}\n")

# 2) Generation quality on the untouched model
def generate(model, prompt, max_new_tokens=40):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,  # GPT-2 has no pad token
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompts = ["Maayong buntag sa tanan.", "Ang mga bata nagdula sa"]
for p in prompts:
    print(generate(model, p), "\n")

