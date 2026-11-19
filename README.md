# Teaching DistilGPT2 Cebuano

Fine-tuning, decoding, and alignment for Cebuano (Bisaya) with **DistilGPT2** — designed to run on a laptop without a GPU.

## Project goal

DistilGPT2 was pretrained mostly on English, so it has little knowledge of Cebuano. This project follows the post-pretraining lifecycle of a language model:

1. Establish what the untouched model can and cannot do.
2. Adapt it to Cebuano with parameter-efficient LoRA fine-tuning.
3. Compare text-generation and inference strategies.
4. Align one clearly defined behavior with Direct Preference Optimization (DPO).
5. Relate DistilGPT2's GPT-2 architecture to modern alternatives.

This is an adaptation project, not pretraining a new model from scratch. The expected result is short, Cebuano-flavored continuations—not a production chatbot or reliable question-answering system.

## Current repository contents

| File | Purpose |
| --- | --- |
| `baseline.py` | Step 0 tokenizer-fragmentation and raw-generation diagnostic |
| `corpus_and_lora.py` | Loads the two Hugging Face datasets and writes `cebuano_corpus.txt` |
| `cebuano_corpus.txt` | Prepared Cebuano training text |
| `finetune_with_lora.py` | LoRA training, loss curve, held-out evaluation, and adapter checkpoint |

## Roadmap

- [x] **Step 0 — Baseline diagnostic:** compare Cebuano and English tokenization and record raw DistilGPT2 generations.
- [x] **Step 1 preparation — Corpus and LoRA pipeline:** reconstruct the corpus and provide a CPU-friendly LoRA training script.
- [x] **Step 1 results:** run training, save `loss_curve.png`, record final loss/perplexity, and compare before versus after fine-tuning.
- [ ] **Step 2 — Decoding and inference:** compare greedy, top-k, top-p/nucleus, and beam search on the same prompts; benchmark `use_cache=True` versus `False`.
- [ ] **Step 3 — DPO alignment:** author at least 30 original Cebuano `prompt,chosen,rejected` pairs for one behavior, train from the LoRA checkpoint, and show the behavior shift.
- [ ] **Step 4 — Architecture note:** explain learned absolute positions versus RoPE or ALiBi, and decoder-only versus encoder-only models for Cebuano tasks.
- [ ] **Final submission:** assemble the report, Canva defense deck, reproducible artifacts, and incremental Git history.

## Data

The corpus combines two public Hugging Face datasets with different Cebuano registers:

- [`josephimperial/CebuaNER`](https://huggingface.co/datasets/josephimperial/CebuaNER): news text reconstructed by joining article tokens and discarding NER labels. License: **CC BY-NC 4.0**.
- [`filbench/cebuano-readability`](https://huggingface.co/datasets/filbench/cebuano-readability): literary and pedagogical passages used as plain text. License: **CC BY 4.0**.

Cite both datasets in the report. The non-commercial license on CebuaNER means this classroom work should not be repurposed commercially.

## Environment setup

```bash
pip install torch transformers datasets peft trl accelerate matplotlib
```

Recommended Python: 3.9 or newer. DistilGPT2 has 82M parameters and can run on CPU; 8 GB or more of RAM is recommended. A GPU is optional and only reduces training time.

## Running the current pipeline

### 1. Prepare the corpus

```bash
python corpus_and_lora.py
```

This downloads both datasets through `datasets`, reconstructs the CebuaNER text, appends the readability passages, and writes `cebuano_corpus.txt`.

### 2. Run the baseline

```bash
python baseline.py
```

Save the token counts, token pieces, prompts, and generated outputs. These are the required “before” measurements.

### 3. Fine-tune with LoRA

```bash
python finetune_with_lora.py
```

The script uses a causal-language-modeling objective with fixed 128-token blocks, holds out 5% of the data, plots training loss, reports held-out perplexity, and saves the adapter under `cebuano_lora_adapter/`.

Before the final run, record the actual trainable-parameter count and use a reproducible seed. Keep the adapter checkpoint: it is the starting point for DPO.

## Required analysis

### Step 0: baseline

Document:

- how many BPE tokens common Cebuano words require compared with similar English words;
- whether raw generations stay in Cebuano, drift into English, fragment, or repeat;
- the exact prompts and decoding settings used.

### Step 1: LoRA adaptation

Report the training configuration, trainable parameters, loss curve, final loss, and optional held-out perplexity. Re-run the Step 0 prompts and compare the raw and adapted models. LoRA changes a small adapter rather than all 82M pretrained weights, which is why this is feasible on a laptop.

The written discussion should connect the result to scaling-law work such as Kaplan et al. (2020) and Hoffmann et al. (2022, “Chinchilla”).

### Step 2: decoding and KV cache

Use the same 3–5 Cebuano prompts for:

- greedy decoding;
- top-k sampling;
- nucleus/top-p sampling;
- beam search.

Compare fluency, repetition, and diversity. Measure generation latency with `use_cache=True` and `use_cache=False`, using the same prompt lengths and generation lengths, and report the speedup rather than only describing it.

### Step 3: DPO alignment

Choose one narrow behavior, such as formal versus casual register or polite versus blunt wording. Author at least 30 original Cebuano preference pairs:

```csv
prompt,chosen,rejected
...
```

The chosen and rejected answers should differ mainly on the selected behavior, remain on-topic, and be reviewed by a Cebuano speaker where possible. Explain the pair-design decisions and how DPO's direct preference loss differs from reward-model-plus-PPO RLHF.

### Step 4: architecture comparison

Explain DistilGPT2's learned absolute positional embeddings and decoder-only design. Compare one newer positional method such as RoPE or ALiBi, then compare decoder-only generation with an encoder-only model such as mBERT or a Filipino BERT variant for tasks like named-entity recognition.

## Deliverables

- **GitHub repository:** scripts/notebook, corpus-preparation code, training logs, loss curve, before/after generations, decoding benchmark, DPO CSV and rationale, and architecture note.
- **Report:** 3–5 pages covering Steps 1–4 with citations.
- **Canva defense deck:** baseline evidence, LoRA results, decoding/KV-cache benchmark, DPO behavior shift, and architecture takeaways.

Commit after each step (`step-0`, `step-1`, etc.) rather than uploading everything in one final commit. This history is part of the assessment.

## Evaluation expectations

Use the same prompts before and after each stage. The project is working when fine-tuning produces more Cebuano-shaped continuations than the baseline and DPO produces a visible shift on the chosen behavior. Do not claim that the model learned reliable conversation: raw-text fine-tuning teaches continuation, not question answering.

Useful lightweight measurements are:

- training loss and held-out perplexity;
- qualitative before/after generations;
- decoding latency and KV-cache speedup;
- a small human preference count on held-out prompts.

BLEU/ROUGE and a full LLM-as-judge pipeline are not required for this project.

## Important caveats

- Ask a native Cebuano speaker to sanity-check generated text; illustrative Cebuano examples are not automatically linguistically correct.
- Save checkpoints outside temporary notebook runtimes.
- Keep DPO focused on one behavior instead of mixing several preferences.
- The optional `saillab/alpaca-cebuano-cleaned` dataset can support instruction fine-tuning, but it is machine-translated and is outside the required scope.

## References

- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2).
- Sanh et al., *DistilBERT, a distilled version of BERT* (distillation background).
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021).
- Kaplan et al., *Scaling Laws for Neural Language Models* (2020).
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022).
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023).
