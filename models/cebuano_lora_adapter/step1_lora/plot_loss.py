import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = r"C:\Users\Erich Adlawan\Downloads\lora_out\lora_out\checkpoint-500\trainer_state.json"
out  = r"C:\Users\Erich Adlawan\Downloads\NLP\loss_curve.png"

with open(path, encoding="utf-8") as f:
    state = json.load(f)

logs = [l for l in state["log_history"] if "loss" in l]
steps = [l["step"] for l in logs]
loss = [l["loss"] for l in logs]

plt.plot(steps, loss)
plt.xlabel("Step"); plt.ylabel("Training loss")
plt.title("LoRA fine-tuning (DistilGPT2, Cebuano)")
plt.grid(alpha=0.3)
plt.savefig(out, dpi=150)

print("Logged points:", len(logs))
print("First loss:", loss[0], "| Final loss:", loss[-1])
print("Steps trained:", state["global_step"])
print("Saved to:", out)
