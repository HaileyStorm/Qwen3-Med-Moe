# Qwen3‑Med‑MoE — K=1+1, N=16

> Dense2MoE‑style conversion of **Qwen3‑1.7B‑Instruct** into a medical‑focused MoE with **one shared expert + top‑1 of 16 normal experts**. Designed as a **drop‑in edge replacement for \~0.6B dense models** (BF16/Q8) with materially stronger medical knowledge and reasoning, while keeping single‑batch latency friendly.

**Canonical model ID:** `Qwen3-Med-MoE-1.7B-K1p1-N16`\
**GitHub Repo:** `HaileyStorm/qwen3-med-moe`\
**Planned HF model:** `HaileyStorm/Qwen3-Med-MoE-1.7B-K1p1-N16`

---

## Why this exists

- **Capacity without latency.** MoE increases *stored* parameters but keeps *active* compute small (always‑on shared + top‑1 normal expert).
- **Edge‑first.** Targets laptops/phones as a replacement for \~0.6B dense deployments (chosen for speed), with higher medical accuracy and reasoning.
- **Dense2MoE → LLM.** We adapt Dense2MoE (originally diffusion/image) to LLM next‑token training: convert **FFNs only** to MoE; **per‑token routing**; staged **KD → CPT → SFT → PO → QAT**.
- **Quantization‑first.** Ship **Q5** for experts+shared; optionally **attention V/O at Q8** late in QAT if eval deltas are acceptable.

> Dense2MoE uses **Taylor‑style neuron importance** and assigns highest‑importance neurons to the **shared expert**. We compute importances over the **full Phase‑0 corpus** (streaming EMA) and port the process to per‑token routed LLMs.

---

## At a glance

- **Base:** Qwen3‑1.7B‑Instruct (hidden 2048, FFN \~6144, 28 layers, SwiGLU)
- **MoE per‑FFN:** **shared 1216** + **N=16 normals 512**, **top‑1** routing
- **Active params:** ≈ **0.94B** (stored ≈ **2.26B**)
- **Router:** softmax top‑1, lb‑loss 0.01–0.015, z‑loss 1e‑4
- **Train precision:** BF16; **Deploy:** Q5 (experts+shared). Router/LN/LM head/QK stay BF16; **V/O‑Q8** may be enabled late in QAT if it meets thresholds.
- **Capacity configs (deploy):** **1.05** or **1.25**. *We evaluate overflow at 1.0 (dropless), but ship only 1.05 or 1.25.*

---

## Repo layout

```
.
├─ configs/         # deepspeed/megatron + eval YAMLs (per phase)
├─ data/            # raw + processed shards
├─ scripts/         # pipeline drivers & utilities
├─ models/          # teacher + checkpoints
├─ dashboards/      # W&B panel presets
└─ docs/            # RUNBOOK, model card drafts
```

**Key scripts**

- `download_data.py` – OA pulls (PubMed/PMC, guidelines, etc.) + license manifests
- `prepare_data.py` – clean, dedup (MinHash/SimHash), **PII scrub**, language filter, domain tags, contamination checks
- `chat_formatting.py` – Qwen template; `/think` markup; rationale strip for `/no_think`
- `pack_sequences.py` – 2k/4k/8k packing with separators & metadata
- `importance_taylor.py` – **full‑corpus** squared‑grad saliency (streaming EMA) → neuron ranking
- `moe_surgery.py` – FFN→MoE (shared=1216; normals=16×512); weight copy; optional k‑means row split; map **highest‑importance → shared** (per Dense2MoE)
- `router_init.py` – shared‑biased init; temp & lb‑loss floors
- `ep_mapping.py` – expert→rank map (**2 normals/GPU**); shared replicated; emits EP config
- `train_stage.py` – phase runner (HF/DeepSpeed/Megatron); **AutoClip** gradient clipping
- `dpo_trainer.py` – SimPO/DPO with candidate sampling via vLLM/HF
- `enable_fake_quant.py` – fake‑quant for **experts+shared**; calibration; persist per‑channel scales
- `qat_stage.py` – staged QAT `Q8→Q6→Q5` with **stop‑anytime** exports; optional **V/O‑Q8** gate
- `eval_vllm.py` – unified eval; **capacity=1.0 dropless**; identical question sets pre‑QAT & final
- `export_hf_ckpt.py` – safetensors + Qwen‑MoE `config.json`; tokenizer; chat template
- `export_awq_gptq.py` – server Q5 (MoE experts/shared). **Dense references** via **calibration‑only PTQ** using the **same mask** as shipped MoE
- `export_gguf_q5.py` – edge GGUF Q5
- `dashboards.py` – create W&B projects/panels (router entropy/usage, token‑drop, lb/z‑loss, MFU, toks/s, evals)
- `autotune_circuit_breakers.py` – metric‑driven LR/capacity/lb‑loss/temp adjustments; rollback checkpoints

---

## Quickstart (eval only)

```bash
git clone https://github.com/HaileyStorm/qwen3-med-moe.git
cd qwen3-med-moe
pip install -r requirements.txt
# download released checkpoints (HF) into models/
python scripts/eval_vllm.py --config configs/eval.yaml \
  --models moe_q5,moe_bf16,1p7b_ptq_mask,0p6b_asserved,0p6b_ptq_mask
```

---

## Data (OA/licensed)

- **Medical raw:** PubMed titles/abstracts; PMC Open‑Access full text; guidelines (CDC/NIH/WHO/NICE); radiology reports (text‑only OA mirrors); ICD‑10‑CM lookups
- **Medical SFT/QA:** MedQA (USMLE), MedMCQA, PubMedQA; curated/synthetic clinical cases; drug label Q&A
- **Non‑medical stabilizers (small):** DCLM‑Baseline, OASST2, no\_robots

**Preprocess (order)**

```bash
python scripts/download_data.py --out data/raw
python scripts/prepare_data.py  --in data/raw --out data/processed
python scripts/chat_formatting.py --in data/processed --out data/processed/chat
python scripts/pack_sequences.py   --in data/processed/chat --out data/processed/packed/phase{0,1,2,3}
```

*Packing lengths:* P0/P1: 4096; P2: mostly 2048 with \~25% 8192; P3: 2048–4096.\
*/think discipline:* default `/no_think`; allow `<think>…</think>` on \~15% hard med‑reasoning items; never distill teacher CoT.

---

## Training (phases)

> **Global token batch/step** (use grad‑accum to reach): **P0 1.5M**, **P1 2.0M**, **P2 1.5M**, **P3 1.0M**, **P4 1.0M**.\
> AdamW (β1=0.9, β2=0.95), wd=0.05 (QAT up to 0.08), cosine LR (3–5% warmup), **AutoClip**.

### P0 — Knowledge Distillation (≈3.5B tokens)

Teacher: Qwen3‑1.7B (dense). Loss: KL on logits (τ=1) + tiny feature L2 (≤0.1).

```bash
python scripts/train_stage.py --config configs/ds_stage0_kd.yaml \
  --data data/processed/packed/phase0
```

### P1 — Medical CPT (≈18B tokens)

Domain‑biased batches early (MeSH‑like tags), then mix.

### P2 — Heavy SFT (≈30B tokens)

Untie embeddings/LM head at start; freeze embeddings for first 20%.\
Enable **Q8 fake‑quant** on **experts+shared** for final **15%**.

```bash
python scripts/enable_fake_quant.py --stage q8 --modules experts,shared
python scripts/train_stage.py --config configs/ds_stage2_sft.yaml --q8_fraction 0.15
```

### P3 — Preference Optimization (≈3.5B tokens)

SimPO/DPO (β=0.1), 3–4 candidates/prompt (vLLM/HF).

### P4 — Quantization‑Aware Training (QAT → Q5; ≤3.0B tokens)

Stages: `Q8 0.1B → Q6 0.1B → Q5 0.3B` (experts+shared), **stop‑anytime** exports.\
After **0.3B at Q5**, **full eval** → if deltas within thresholds (medical ≥ −1.0 pp; non‑medical ≥ −0.5 pp, no safety regressions), enable **attention V/O‑Q8** for **+0.1B**, re‑eval at **0.4B**, then continue with/without V/O‑Q8 accordingly.\
Checkpoints + subset eval every **0.1B**; full evals at **0.3B**, **0.4B**, then every **0.2B**.

```bash
python scripts/qat_stage.py --stages q8:0.1b,q6:0.1b,q5:0.3b \
  --stop_anytime --export_each_stage
```

---

## Evaluation

**Models/precisions**

- **MoE:** final **Q5**, plus **BF16 pre‑QAT** on the *same* question sets (sanity)
- **Qwen3‑1.7B (dense):** PTQ with the **same mask as shipped MoE** (FFN‑Q5; or FFN‑Q5 + V/O‑Q8 if MoE ships that). **No QAT.**
- **Qwen3‑0.6B (dense):** (1) **as‑served baseline** (BF16 or Q8), and (2) PTQ with the **same mask as shipped MoE**. **No QAT.**

**/think usage:** default `/no_think`; run `/think` ablations on MedQA‑hard, clinical cases, GSM8K‑mini.\
**Runtime:** vLLM; **capacity=1.0 dropless** for *all* evals (fair apples‑to‑apples).\
**Artifacting:** CSVs, confusion matrices, W&B tables; latency/tokens‑per‑sec.

Run:

```bash
python scripts/eval_vllm.py --config configs/eval.yaml \
  --models moe_q5,moe_bf16,1p7b_ptq_mask,0p6b_asserved,0p6b_ptq_mask
```

---

## Monitoring & circuit breakers

**W&B projects:** `moe-phase0`, `moe-phase1`, `moe-phase2`, `moe-phase3`, `moe-qat`, `moe-eval`.

**Dashboards:** router usage/entropy, **token‑drop**, lb/z‑loss; loss, med/general PPL, toks/s, MFU; task accuracies.\
**Breakers (auto & logged):** overflow >0.5% → capacity +0.05 & temp −0.05; collapse (>40% one expert) → lb +0.002 & temp −0.05; under‑utilization → temp +0.05 & soft noise; QAT regress (>1.5 pp vs best BF16 for 3 evals) → rollback stage, +0.05B tokens, per‑block recalib.

---

## Packaging & deployment

- **HF safetensors** + Qwen‑MoE `config.json`; tokenizer + chat template
- **Server:** Q5 AWQ/GPTQ (experts/shared)
- **Edge:** Q5 GGUF
- **Capacity configs:** ship **1.05** and **1.25** (evaluate overflow at **1.0 dropless**).
- vLLM compatibility (Qwen3‑Next MoE config surface)

---

## License

- **Code (this repo):** Apache‑2.0. See `LICENSE`.
- **Model weights:** Derived from **Qwen3‑1.7B‑Instruct**, which is released under **Apache‑2.0** on Hugging Face. We redistribute fine‑tuned/converted weights under **Apache‑2.0** with attribution and include upstream notices; see `NOTICE`.
- **Datasets:** Not redistributed here. We trained on OA/licensed sources; users are responsible for complying with each dataset’s license/terms if they reproduce the pipeline.

**Legal notes.** Preserve upstream notices in any downstream distribution. This project is **not medical advice**; see Safety section for disclaimers.

---

## References

- **Qwen3‑1.7B‑Instruct (HF, Apache‑2.0):** [https://huggingface.co/Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- **vLLM Qwen3‑Next MoE config:** [https://docs.vllm.ai/en/latest/api/vllm/transformers\_utils/configs/qwen3\_next.html](https://docs.vllm.ai/en/latest/api/vllm/transformers_utils/configs/qwen3_next.html)
- **Dense2MoE (Taylor‑importance init; our LLM adaptation):** [https://arxiv.org/html/2510.09094v1](https://arxiv.org/html/2510.09094v1)
- **Qwen‑MoE background blog:** [https://qwenlm.github.io/blog/qwen-moe/](https://qwenlm.github.io/blog/qwen-moe/)
- **Datasets:** PubMed — [https://huggingface.co/datasets/ncbi/pubmed](https://huggingface.co/datasets/ncbi/pubmed) • PMC OA — [https://pmc.ncbi.nlm.nih.gov/tools/openftlist/](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/) • MedQA — [https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf) • MedMCQA — [https://huggingface.co/datasets/openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) • PubMedQA — [https://huggingface.co/datasets/llamafactory/PubMedQA](https://huggingface.co/datasets/llamafactory/PubMedQA) • MIMIC‑CXR reports — [https://huggingface.co/datasets/tgrex6/mimic-cxr-reports-summarization](https://huggingface.co/datasets/tgrex6/mimic-cxr-reports-summarization) • DCLM‑Baseline — [https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) • OASST2 — [https://huggingface.co/datasets/OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) • no\_robots — [https://huggingface.co/datasets/HuggingFaceH4/no\_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)

