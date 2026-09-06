# Inference Recipes

Measured, reproducible recipes for serving open models on rented GPUs.

## Completed

### RTX 5090 · Qwen3.8-27B NVFP4 · SGLang

- [Full report](5090/qwen3.8-27b-nvfp4/sglang/REPORT.md)
- [Summary CSV](5090/qwen3.8-27b-nvfp4/sglang/summary.csv)
- [Concurrency CSV](5090/qwen3.8-27b-nvfp4/sglang/concurrency.csv)
- [Context/output limits CSV](5090/qwen3.8-27b-nvfp4/sglang/limits.csv)
- [GPU telemetry summary](5090/qwen3.8-27b-nvfp4/sglang/telemetry.csv)
- [Raw artifacts](5090/qwen3.8-27b-nvfp4/sglang/artifacts/2026-09-06-vast-rtx5090/)

Headline results:

- DFlash2: **188.55 output tok/s** at 1K input / 256 output.
- MTP-5: **161.25 output tok/s** at 1K input / 256 output.
- MTP-5 concurrency 8: **646.05 aggregate output tok/s**.
- MTP-5 validated: **1,024 input + 41,300 output tokens**.
- Max-context mode validated: **260,000 input + 1,024 output tokens**.

The report contains the complete readable `sglang serve` commands. Charts are generated locally with:

```bash
python3 5090/qwen3.8-27b-nvfp4/sglang/analyze_results.py
```

## Planned

- Explicit FlashInfer CUTLASS vs cuDNN vs CuTe DSL NVFP4 GEMM comparison
- DFlash2 concurrency and coding-trace acceptance
- vLLM comparison
- RTX PRO 6000
- DGX Spark / GB10
