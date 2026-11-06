| Step | What You Do | Main Output |
|---|---|---|
| 0 | Run stock `distilgpt2` on Cebuano and show it fails. | Token-count table and bad generations |
| 1 | Build the corpus, then LoRA fine-tune. | Loss curve, before/after samples, saved adapter |
| 2 | Generate with greedy, top-k, top-p, and beam; time `use_cache` on and off. | Comparison table and speedup number |
| 3 | Write 30+ Cebuano preference pairs on ONE behavior, then run DPO. | CSV, before/after samples, win rate |
| 4 | Write about RoPE/ALiBi and encoder vs. decoder. | Writing only |
