# Adapting Dense2MoE to create Qwen3-Med-MoE-1.7B-K1p1-N16: Project Plan

Aliases:

- **Model ID (canonical):** `Qwen3-Med-MoE-1.7B-K1p1-N16`
- **GitHub Repo:** `HaileyStorm/Qwen3-Med-Moe`
- **HuggingFace hub:** `HaileyStorm/Qwen3-Med-MoE-1.7B-K1p1-N16`

---

## 0) Motivation & Scope (self‑contained)

**Objective.** Convert **Qwen3‑1.7B‑Instruct** (dense) into a compact **Mixture‑of‑Experts (MoE)** model with **K=1+1** (one always‑on **shared** expert plus top‑1 **normal** expert) and **N=16** normal experts, specialized for **medical conversation & QA**, while keeping **active parameters ≤ \~0.95B**. Target broad deployment (laptops/phones) with **Q5** weight‑only quantization for **experts + shared FFN**.

**Designed as a drop‑in replacement for BF16 or Q8 \~0.6B edge models.** The project explicitly aims to **replace \~0.6B dense deployments** (chosen for **edge speed**). We target **clearly better medical performance** and **higher reasoning ability** versus the **0.6B** baseline, along with an increased knowledge base, all while keeping single‑batch latency competitive on laptops/phones. Against **1.7B dense**, we expect **small to moderate medical gains** with **general performance near parity**.

**Why MoE at this scale.** Small dense models have a tight budget to retain broad capabilities plus deep domain knowledge. MoE expands **stored capacity** (many parameters) while keeping **activated compute** per token low (only shared + one normal expert run). That lets us store more specialized medical knowledge without hurting single‑batch latency, likely with better general performance.

**Method lineage & adaptation.** We draw on the ideas in **Dense2MoE** (convert dense FFNs to experts; staged repair/finetune; load balancing). Dense2MoE was shown on a diffusion image generation transformer model; we **adapt** the recipe to autoregressive LLMs:

- Convert **FFNs only** to MoE; keep attention dense.
- Use **per‑token routing** (not denoise steps), with top‑1 selection among normal experts.
- Stage the training: **P0 KD → P1 med CPT → P2 heavy SFT → P3 DPO/SimPO → P4 QAT**.
- Initialize experts using **Taylor‑style neuron importance**, **assigning the highest‑importance neurons to the shared expert** (as in per Dense2MoE). Our extension is computing importances over the **full Phase‑0 corpus** via **streaming EMA**.
- Use **/no\_think** by default; restrict `/think` to hard medical reasoning (where data available for training), and never distill teacher CoT text.

**What’s novel here (beyond “a medical model”):**

- A practical, deployable **K=1+1, N=16** MoE conversion of Qwen3 with staged **QAT→Q5**, shipped in **HF safetensors** & **GGUF**, using a **stock Qwen MoE config** (vLLM‑compatible) and minimal custom code.
- The **adaptation of Dense2MoE from diffusion/image to LLM** shows a **cost‑effective path** to create new, (optionally) domain‑specialized models from strong dense bases.
- A **reproducible pipeline** with automation & circuit breakers for router stability and quant robustness; all **scripts, configs, dashboards, evals** will be published.

---

## 1) Architecture & Prompting

### 1.1 Source & Target

- **Source (teacher & initializer):** `Qwen3‑1.7B‑Instruct` (hidden 2048, FFN \~6144, 28 layers, SwiGLU).
- **Target (student):** Per FFN block → **MoE module**:\
  **Shared expert** width **1216** (\~r\_s=0.198) + **N=16 normal experts** width **512** (\~r\_n=0.0833).\
  **Router:** softmax, **top‑1**, load‑balance aux‑loss (0.01→0.015), **z‑loss 1e‑4**, **capacity factor 1.25** during training.
- **Parameter accounting:** **Active ≈ 0.94B**, **stored ≈ 2.26B**.

### 1.2 Why K=1+1 (and no Mixture‑of‑Blocks in v1)

- K=1+1 keeps routing simple/robust and minimizes bandwidth; K>1 destabilizes small models and increases latency. It would also necessitate more, smaller experts in order to match VRAM, and the chosen experts are already quite small.
- Mixture‑of‑Blocks can be explored in v2; it complicates surgery and checkpointing without being necessary for v1’s goals.

### 1.3 Prompting & `/think`

- Use Qwen chat template.  Default `/no_think` allow `/think` on **\~15%** of hard med‑reasoning samples. Put rationales inside `<think>…</think>`. Strip rationales for `/no_think` runs. We **never** distill teacher CoT.

### 1.4 Untying embeddings/LM head

- Keep tied through P0/P1 for stability. **Untie at the start of P2 (Heavy SFT)**. **Freeze embeddings for the first 20% of P2**, then unfreeze if metrics improve.

### 1.5 Precision policy (train vs deploy)

- Train in **BF16**, quantize **experts + shared FFN**, and *(optionally)* **attention V/O at Q8 only in late P4** (see §3 P4). Keep the remaining attention, routers, LayerNorms, and LM head **BF16**. Deploy **Q5** on experts/shared (server Q5 AWQ/GPTQ; edge Q5 GGUF).

### 1.6 Baseline we replace & expected reasoning delta

- We are designing this to **replace \~0.6B dense models** used on edge for their speed. With **N=16** specialization and larger stored capacity, we **expect improved reasoning** vs **0.6B**, improved performance general and particularly medical tasks; against **1.7B dense**, we expect **medical gains** with general parity.

## 2) Data Plan (sources, splits, packing, governance) (sources, splits, packing, governance)

### 2.1 Datasets (medical)

- **PubMed (titles/abstracts):** [https://huggingface.co/datasets/ncbi/pubmed](https://huggingface.co/datasets/ncbi/pubmed)
- **PMC Open‑Access subset (full text, OA only):** [https://pmc.ncbi.nlm.nih.gov/tools/openftlist/](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/) : [https://huggingface.co/datasets/ncbi/pubmed](https://huggingface.co/datasets/ncbi/pubmed)]\([https://huggingface.co/datasets/ncbi/pubmed](https://huggingface.co/datasets/ncbi/pubmed))
- **PMC Open‑Access subs**[et](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)[ (full text, OA only): ](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)[https://pmc.ncbi.nlm.nih.gov/tools/openftlist/](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)
- **Clinical guidelines** (OA): CDC, NIH, WHO, NICE pages (crawl → text with license fields).
- **Radiology reports** (text only): e.g., HF mirrors of MIMIC‑CXR reports (verify license): [https://huggingface.co/datasets/tgrex6/mimic-cxr-reports-summarization](https://huggingface.co/datasets/tgrex6/mimic-cxr-reports-summarization)
- [**MedQA (USMLE)**](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf)[: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf)
- [**MedMCQA**](https://huggingface.co/datasets/openlifescienceai/medmcqa)[: https://huggingface.co/datasets/openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa)
- [**PubMedQA**](https://huggingface.co/datasets/llamafactory/PubMedQA)[: https://huggingface.co/datasets/llamafactory/PubMedQA](https://huggingface.co/datasets/llamafactory/PubMedQA)
- **Clinical cases** (curated & synthetic): internal generation templates; enforce structure (CC, HPI, PMH, meds/allergies, vitals, labs/imaging, assessment/plan) + rationale (if `/think`).
- **Drug labels (OA)**: DailyMed FDA labels (XML→text), build Q&A (contraindications, dosing, interactions).
- **ICD‑10‑CM** public lists (coding lookups & prompts).

### 2.2 Datasets (non‑medical stabilizers, small)

- **DCLM-Baseline**: [https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0 ](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0 )
- [**OpenAssistant/oasst2**](https://huggingface.co/datasets/OpenAssistant/oasst2)[: ](https://huggingface.co/datasets/OpenAssistant/oasst2)[https://huggingface.co/datasets/OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2)
- [**HuggingFaceH4/no\_robots**](https://huggingface.co/datasets/HuggingFaceH4/no_robots)[: ](https://huggingface.co/datasets/HuggingFaceH4/no_robots)[https://huggingface.co/datasets/HuggingFaceH4/no\_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)

### 2.3 Splits & contamination control

- Create **train/dev/test** splits (80/10/10) where not provided; store manifests with SHA256, license, and URL.
- Run **near‑dup** detection (SimHash/MinHash); remove collisions with eval sets.
- Keep **eval test sets** completely unseen.

### 2.4 Packing & lengths

- **P0/P1:** seq **4096**; **P2:** mostly **2048**, with **\~25% 8192** for long cases; **P3** pairs **2048–4096**.
- **/think tagging:** apply to marked hard items; strip any CoT when `/no_think`.

### 2.5 Preprocessing scripts & order

1. `download_data.py` → `data/raw/`
2. `prepare_data.py` → clean, dedup, **PII scrub**, language filter, MeSH‑like **domain tagging**, contamination audit → `data/processed/`
3. `chat_formatting.py` → Qwen chat template, `/think` & `<think>` markup, rationale stripping → `data/processed/chat/`
4. `pack_sequences.py` → 2k/4k/8k packing with EOS and metadata retention → `data/processed/packed/{phase}`

---

## 3) Training Schedule & Hyperparameters

> **Global token batch per step** (use grad accumulation to reach):\
> **P0 1.5M** • **P1 2.0M** • **P2 1.5M** • **P3 1.0M** • **P4 (QAT) 1.0M**.

**Common:** AdamW (β1=0.9, β2=0.95), weight decay **0.05** (allow **0.08** during QAT if neutral), cosine LR (warmup 3–5%), grad clip **"AutoClip"** style setting clip to 10th percentile of rolling grad norm window, BF16 compute/FP32 master, activation checkpointing. Router: top‑1, **lb‑loss 0.01–0.015**, **z‑loss 1e‑4**, **capacity 1.25** (training); temperature anneal (e.g., 0.9→0.6 P0/P1; 0.5→0.4 P2).

### P0 — Knowledge Distillation (3.5B tokens)

- **Purpose:** align token distributions; warm‑stabilize router.
- **Teacher:** Qwen3‑1.7B‑Instruct (dense).
- **Loss:** KL on logits (τ=1), optional small feature L2 on last hidden (≤0.1).
- **Seq:** 4096.
- **LR:** 3e‑5 → 1e‑5.
- **Init:** router bias toward shared; temp 0.9; capacity 1.25.
- **Data:** \~60% med raw / 40% general raw (no CoT).

### P1 — Medical CPT (18B tokens)

- **Mix:** 80–85% med raw, 10–15% med SFT (format adapters), small general raw.
- **Seq:** 4096.
- **LR:** 1.5e‑5 → 1e‑6.
- **Regularizers:** expert‑dropout 0.05, lb‑loss 0.015→0.01, temp 0.9→0.6.
- **Scheduling:** first \~1/3 with **domain‑biased batches** (by MeSH‑like tags), then mix.

### P2 — Heavy SFT (30B tokens)

- **Untie embeddings/LM head** at start; **freeze embeddings for first 20% of P2**.
- **Seq:** mostly 2048, with \~25% 8192 for long cases.
- **LR:** 8e‑6 → 5e‑7.
- **Router:** temp 0.5→0.4; lb‑loss ≈0.01.
- **Quant robustness:** enable **Q8 fake‑quant** on **experts + shared FFN** for final **15% of steps** (`enable_fake_quant.py`).

### P3 — Preference Optimization (3.5B tokens)

- **Method:** DPO/SimPO, β=0.1; **3–4 candidates/prompt** (sampled via vLLM/HF `generate`).
- **Focus:** correctness & safety; uncertainty phrasing; concise but complete plans.

### P4 — Quantization‑Aware Training (QAT → Q5)

- **Stages:** `Q8 0.1B → Q6 0.1B → Q5 0.3B` tokens on **experts + shared**.
- **Gate for attention V/O Q8:** After **0.3B at Q5**, run a **full eval**. If deltas vs best BF16‑preQAT are within thresholds (**medical ≥ −1.0 pp**, **non‑medical ≥ −0.5 pp**, no safety regressions), enable **attention V/O at Q8** and train **+0.1B** tokens; run a **full eval** at **0.4B**. Continue either **with** V/O‑Q8 (if thresholds maintain) **or without** (revert) for the rest of P4.
- **Cap & cadence:** Keep **P4 ≤ 3.0B tokens** total; **checkpoint + subset eval every 0.1B**; **full evals** at **0.3B**, **0.4B**, then every **0.2B**.
- **Scope:** quantize **experts + shared** (per‑channel row‑wise scales, group 64–128, learnable clip); keep **Q/K**, router, LN, LM head **BF16** (only **V/O** may be Q8 as above).
- **Calibration:** short pass before each stage.
- **LR/decay tweaks:** slightly higher LR early per stage; weight decay up to 0.08 if neutral; maintain **EMA/Polyak** from late P2 through QAT.
- **Stop‑anytime:** export real Q weights at the end of **each stage** and every **0.1B** tokens in Q5.

**Optional branch (off by default):** after finishing the 15% Q8 steps in P2, **repeat that subset in BF16** and run a tiny DPO fork to compare against the Q8‑touched path. Include only if analyzing quant noise during SFT.

---

## 4) Evaluation Suite & Protocols

**Models & precisions:**

- **MoE:** final **Q5**; also **BF16 pre‑QAT** on the *same* question sets (sanity regression check).
- **Qwen3‑1.7B (reference, dense):** **Quantized baseline = calibration‑only PTQ** applying the **exact same quantization mask** as the shipped MoE (i.e., **FFN‑only Q5** if MoE ships FFN‑only; **FFN‑Q5 + attention V/O‑Q8** if MoE ships with V/O‑Q8). **No QAT** for references.
- **Qwen3‑0.6B (reference, dense):** Evaluate **two**: (1) the **as‑served baseline** (**BF16 or Q8**, unchanged), and (2) the **quantized baseline = calibration‑only PTQ** with the **same mask as the shipped MoE** (as above). **No QAT** for references.

**Baseline quantization policy (calibration‑only):** Reference models **mirror the MoE’s shipped quantization mask** using standard PTQ calibration; **no optional variants** and **no QAT**.

**/think handling:** default `/no_think`; run **/think ablations** on MedQA‑hard, clinical cases, GSM8K‑mini.

**Runtime:** vLLM with Qwen3‑Next/MoE settings; **capacity=1.0 dropless** for *all* evals; greedy decode for MCQ; fixed few‑shot prompts; record latency/tokens/s.

**Artifacts:** CSVs, plots, confusion matrices, and W&B tables.

---

## 5) Implementation Plan (repo, scripts, local tests, cluster runs)

### 5.1 Repository layout

```
repo/
  configs/        # deepspeed/megatron + eval YAMLs per stage
  data/           # raw + processed shards
  scripts/        # drivers/utilities (below)
  models/         # teacher + checkpoints
  dashboards/     # W&B presets
  docs/           # README, RUNBOOK, model card drafts
```

### 5.2 Scripts (must exist before cluster time)

- `download_data.py` — HF pulls; PMC OA list → crawl; guidelines fetch; license manifests.
- `prepare_data.py` — clean, dedup (MinHash/SimHash), language filter, **PII scrub**, MeSH‑like tags, contamination checks.
- `chat_formatting.py` — Qwen formatting; `/think` markup; rationale strip for `/no_think`.
- `pack_sequences.py` — 2k/4k/8k packing with separators & metadata.
- `importance_taylor.py` — **full‑corpus** squared‑grad saliency (streaming EMA) to rank FFN neurons (extension beyond small‑shard practice).
- `moe_surgery.py` — FFN→MoE with (shared=1216, normals=16×512), weight copy/scale, optional k‑means row split; ensures **highest‑importance neurons are mapped to shared** (per Dense2MoE) by design.
- `router_init.py` — bias toward shared; set temp & lb‑loss floors.
- `ep_mapping.py` — map **2 normals/GPU**; shared replicated; emits DS/Megatron EP conf.
- `train_stage.py` — generic runner (HF/DeepSpeed/Megatron) by phase YAML; logs.
- `dpo_trainer.py` — SimPO/DPO with candidate sampling via vLLM/HF; pair assembly.
- `enable_fake_quant.py` — wrap **experts+shared** with fake‑quant; **calibration** passes; persist per‑channel scales.
- `qat_stage.py` — Q8→Q6→Q5 staged loop; **stop‑anytime**; export real Q.
- `eval_vllm.py` — unified eval; identical question sets pre‑QAT and final; capacity=1.0 dropless (serve 1.05 or 1.25 depending on result of capacity overflow eval).
- `export_hf_ckpt.py` — HF safetensors + Qwen‑MoE `config.json`; tokenizer & chat template.
- `export_awq_gptq.py` — server Q5 packaging (experts/shared only for MoE). **Also supports dense references** via **calibration‑only PTQ** driven by a **mask file that mirrors the MoE’s shipped quantization mask** (FFN‑only Q5 or FFN‑Q5 + attention V/O‑Q8). No QAT.
- `export_gguf_q5.py` — edge Q5 packaging.
- `dashboards.py` — W&B project creation per phase; router/perf/eval panels.
- `autotune_circuit_breakers.py` — metric‑driven LR/capacity/lb‑loss/temp tweaks; rollback.

### 5.2.1 Pre‑flight checklist (local, before cluster time)

- Lock **GitHub repo name**: `qwen3-med-moe` (GitHub **only**); keep model ID as `Qwen3‑Med‑MoE‑1.7B‑K1p1‑N16`.
- Implement & unit‑test **every script** above on small synthetic JSONL data.
- Run **end‑to‑end local dry‑run**: `download→prepare→chat_format→pack→importance→surgery→router_init→eval(mini)`.
- Verify **determinism**: set seeds for `torch`, `numpy`, `random`; pin cuBLAS workspace; log seeds in W&B.
- Confirm **DataLoader throughput** with Arrow mmap, workers/prefetch settings; profile CPU bottlenecks.
- Write **phase YAMLs** and sanity‑try `train_stage.py` on a toy 2‑layer model; confirm router stats logging.
- Validate **QAT wrappers**: fake‑quant forward, per‑channel scales, export path round‑trips.
- Prepare **evaluation question sets** (final lists), so pre‑QAT = final sets (selected subset actually run pre‑QAT).

### 5.3 Local dry‑runs (before 8×H100)

- **Unit tests** for each script (small synthetic JSONL).
- **Mini‑surgery** on Qwen3-0.6B-Instruct; verify shapes, forward pass, router outputs.
- **Dataloader soak**: pack & stream 50–100GB locally with DataLoader (see §5.5).
- **Eval harness** on 100‑prompt slices; verify scoring & outputs.
- **QAT wrappers**: run fake‑quant on a tiny subset; confirm scale learning & export.

### 5.4 Cluster bring‑up (8×H100)

- Confirm NCCL/InfiniBand, NVLink topology; set env (NCCL\_ASYNC\_ERROR\_HANDLING, NCCL\_MIN\_NRINGS=4, etc.).
- DeepSpeed/Megatron configs for **EP** (2 normals/GPU), **ZeRO‑3**, activation checkpointing.
- Start with P0 small LR warm‑start; check MFU and toks/s; tune micro‑batch & grad‑accum to hit **global tokens/step** targets.

### 5.5 DataLoader settings (HF/PyTorch)

- `num_workers=6–8` per GPU; `prefetch_factor=4`; `pin_memory=True`; `persistent_workers=True`.
- HF Datasets: memory‑mapped Arrow; **shard by rank**; `set_epoch` per sampler; disable Python GIL hotspots in preprocessing.

### 5.6 Commands (examples)

- **P0:** `python scripts/train_stage.py --config configs/ds_stage0_kd.yaml --data data/processed/phase0`
- **P2 with Q8 steps:**
  - `python scripts/enable_fake_quant.py --stage q8 --modules experts,shared`
  - `python scripts/train_stage.py --config configs/ds_stage2_sft.yaml --q8_fraction 0.15`
- **QAT staged:** `python scripts/qat_stage.py --stages q8:0.1b,q6:0.1b,q5:0.3b --stop_anytime --export_each_stage`

---

## 6) Monitoring, Dashboards, and Circuit Breakers

**W&B projects:** `moe-phase0`, `moe-phase1`, `moe-phase2`, `moe-phase3`, `moe-qat`, `moe-eval`.

- Log **CLI, git SHA, configs**, dataset hashes; upload scripts as artifacts.
- Dashboards:\
  **Router:** per‑expert token share, entropy, **token‑drop rate**, overflow %, **lb‑loss**, **z‑loss**.\
  **Perf:** loss, med/general PPL, toks/s, MFU, long‑context latency.\
  **Eval:** MedQA/MedMCQA/PubMedQA acc; clinical long‑form ROUGE/BERTScore; MMLU‑5 & ARC‑C acc; GSM8K EM.

**Circuit breakers (auto + logged):**

- **Overflow > 0.5%** for >1k steps → **capacity += 0.05**, **temp −= 0.05**.
- **Expert collapse** (top‑1 expert >40% share for >2k steps + entropy↓) → **lb‑loss += 0.002**, **temp −= 0.05**.
- **Under‑utilization** (≥4 experts <3% share) → **temp += 0.05**, inject soft routing noise for 10k steps.
- **Med dev PPL rising** for 3 evals → **LR × 0.5**; if persists, rollback 1 ckpt & resume with domain‑biased batches for 0.2B tokens.
- **QAT regress** (Q5 −1.5pp vs best BF16 for 3 evals) → rollback stage, **+0.05B** stage tokens, per‑block recalib; if still bad, **ship best BF16 pre‑QAT** alongside Q5.

All changes record **reason** and snapshot a checkpoint for revertability.

---

## 7) Packaging & Deliverables

**Deliverables:**

- **Models:** MoE **Q5** (HF safetensors + `config.json`), **Q5 GGUF**; **BF16 pre‑QAT** checkpoint.
- **Configs:** DS/Megatron stage configs; vLLM MoE config; tokenizer; chat template.
- **Evals:** CSVs, plots, W&B dashboards; latency/tokens/s.
- **Code:** scripts & configs in GitHub (`qwen3-med-moe`); HF model card with data summary, `/think` guidance, and safety notes.

**Deploy defaults:** Provide two configs: **capacity=1.05** and **1.25**. We **evaluate overflow at capacity=1.0 dropless** during benchmarking, but **ship only 1.05 or 1.25**. Pick default by overflow: if drops at 1.0 → **ship 1.25**; otherwise **ship 1.05**. Keep top‑1 routing; no external code.

---

## 8) Risks & Mitigations (automated where possible)

- **Router collapse / overflow:** handled by breakers (lb‑loss/temp/capacity) + domain‑biased batching.
- **Overfit to med SFT:** small non‑med stabilizers; track MMLU/ARC; early stop on general PPL degradation.
- **QAT quality loss:** staged QAT with calibration, per‑channel scales, EMA; **stop‑anytime** with export; ship Q5 even if slightly below BF16 (publish both).
- **Licensing/PII:** OA‑only; PII scrub + audits; no CPT/SNOMED.
- **Runtime mismatch:** standard Qwen MoE config; vLLM tested; GGUF for edge.

---

## 9) Timeline

1. **Prep (local)** — Repo, scripts, unit tests, data pulls, packing tests, eval harness sanity.
2. **P0** — KD warm repair; quick eval slice; adjust router knobs.
3. **P1** — Med CPT; periodic evals; breaker tuning.
4. **P2** — Heavy SFT (incl. 15% Q8 steps); pre‑QAT BF16 eval.
5. **P3** — DPO/SimPO; final prompt templates lock.
6. **P4** — QAT staged to Q5; export server/edge; final evals & packaging.

---

## 10) Notes on Quantization Robustness

- Maintain **slightly higher LR** early in each QAT stage; prefer **higher weight decay** (up to 0.08) if neutral.
- Use **EMA/Polyak averaging** in late P2 & throughout QAT.
- Prefer **per‑channel row‑wise** scales, small **group size (64–128)**, and **learnable clipping**.
- Keep **LM head/attention (except perhaps V/O)/router/LN** in BF16.

---

## 11) References

- Qwen3‑1.7B‑Instruct [params (Modular builds): ](https://builds.modular.com/models/Qwen3/1.7B)[https://builds.modular.com/models/Qwen3/1.7B](https://builds.modular.com/models/Qwen3/1.7B)
- vLLM Qwen3‑Next MoE config surface: [https://docs.vllm.ai/en/latest/api/vllm/transformers\_utils/configs/qwen3\_next.html](https://docs.vllm.ai/en/latest/api/vllm/transformers_utils/configs/qwen3_next.html)
- Dense2MoE overview (original uses Taylor‑style neuron importance for expert init; we extend via full‑corpus streaming‑EMA importances and per‑token LLM adaptation): [https://arxiv.org/html/2510.09094v1](https://arxiv.org/html/2510.09094v1)
- Qwen‑MoE blog (background): [https://qwenlm.github.io/blog/qwen-moe/](https://qwenlm.github.io/blog/qwen-moe/)
- [Datasets: PubMed ](https://huggingface.co/datasets/ncbi/pubmed)[https://huggingface.co/datasets/ncbi/pubmed](https://huggingface.co/datasets/ncbi/pubmed)[ • PMC OA ](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)[https://pmc.ncbi.nlm.nih.gov/tools/openftlist/](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)[ • MedQA ](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf)[https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf)[ • MedMCQA ](https://huggingface.co/datasets/openlifescienceai/medmcqa)[https://huggingface.co/datasets/openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa)[ • PubMedQA ](https://huggingface.co/datasets/llamafactory/PubMedQA)[https://huggingface.co/datasets/llamafactory/PubMedQA](https://huggingface.co/datasets/llamafactory/PubMedQA)[ • MIMIC‑CXR reports ](https://huggingface.co/datasets/tgrex6/mimic-cxr-reports-summarization)[https://huggingface.co/datasets/tgrex6/mimic-cxr-reports-summarization](https://huggingface.co/datasets/tgrex6/mimic-cxr-reports-summarization)[ • Datasets: PubMed [https://huggingface.co/datasets/ncbi/pubmed](https://huggingface.co/datasets/ncbi/pubmed) • PMC OA [https://pmc.ncbi.nlm.nih.gov/tools/openftlist/](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/) • MedQA [https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf) • MedMCQA [https://huggingface.co/datasets/openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) • PubMedQA [https://huggingface.co/datasets/llamafactory/PubMedQA](https://huggingface.co/datasets/llamafactory/PubMedQA) • MIMIC‑CXR reports [https://huggingface.co/datasets/tgrex6/mimic-cxr-reports-summarization](https://huggingface.co/datasets/tgrex6/mimic-cxr-reports-summarization) • DCLM‑Baseline [https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) • OASST2 [https://huggingface.co/datasets/OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) • no\_robots [https://huggingface.co/datasets/HuggingFaceH4/no\_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)

