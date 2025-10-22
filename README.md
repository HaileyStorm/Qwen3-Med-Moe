# Qwen3-Med-Moe
An attempt to adapt Dense2MoE to convert Qwen3‑1.7B‑Instruct (dense) into a compact Mixture‑of‑Experts (MoE) model with K=1+1 (one always‑on shared expert plus top‑1 normal expert) and N=16 normal experts, specialized for medical conversation &amp; QA, while keeping active parameters ≤ ~0.95B
