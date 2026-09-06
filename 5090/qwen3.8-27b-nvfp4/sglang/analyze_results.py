#!/usr/bin/env python3
"""Build compact CSV summaries and plots from SGLang benchmark JSONL files."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RAW = HERE / "artifacts" / "2026-09-06-vast-rtx5090" / "qwen38-results"
PLOTS = HERE / "plots"
PLOTS.mkdir(exist_ok=True)


def load(name: str) -> dict:
    return json.loads((RAW / name).read_text().splitlines()[-1])


def write_csv(name: str, rows: list[dict]) -> None:
    with (HERE / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


configs = {
    "AR": "baseline-flashinfer-auto-fp32",
    "MTP-4": "mtp-flashinfer-auto-bf16",
    "MTP-5": "mtp5-flashinfer-auto-bf16",
    "DFlash2": "dflash2-flashinfer-auto-bf16",
}
summary = []
for label, stem in configs.items():
    for workload in ("micro", "official"):
        d = load(f"{stem}.{workload}.jsonl")
        summary.append(
            {
                "config": label,
                "workload": workload,
                "input_tokens": d["random_input_len"],
                "output_tokens": d["random_output_len"],
                "output_tok_s": round(d["output_throughput"], 2),
                "mean_ttft_ms": round(d["mean_ttft_ms"], 2),
                "mean_tpot_ms": round(d["mean_tpot_ms"], 2),
                "p95_tpot_ms": round(d["p95_tpot_ms"], 2),
                "accept_length": round(d.get("accept_length", 0) or 0, 2),
            }
        )
write_csv("summary.csv", summary)

concurrency = []
for c in (1, 2, 4, 8):
    d = load(f"mtp5-throughput-c8.c{c}.jsonl")
    concurrency.append(
        {
            "requested_concurrency": c,
            "measured_concurrency": round(d["concurrency"], 2),
            "output_tok_s": round(d["output_throughput"], 2),
            "request_s": round(d["request_throughput"], 2),
            "mean_ttft_ms": round(d["mean_ttft_ms"], 2),
            "p95_ttft_ms": round(d["p95_ttft_ms"], 2),
            "mean_tpot_ms": round(d["mean_tpot_ms"], 2),
            "accept_length": round(d.get("accept_length", 0) or 0, 2),
        }
    )
write_csv("concurrency.csv", concurrency)

limits = []
for label, file in {
    "DFlash2 48K prompt": "dflash2-flashinfer-auto-bf16.context48k.jsonl",
    "MTP-5 32K output": "mtp5-long-output-32k.jsonl",
    "MTP-5 41.3K output": "mtp5-max-output-41300.jsonl",
    "AR 260K prompt": "max-context-ar-bf16-260k.jsonl",
}.items():
    d = load(file)
    limits.append(
        {
            "test": label,
            "input_tokens": d["total_input_tokens"],
            "output_tokens": d["total_output_tokens"],
            "retokenized_output_tokens": d["total_output_tokens_retokenized"],
            "duration_s": round(d["duration"], 2),
            "output_tok_s": round(d["output_throughput"], 2),
            "ttft_ms": round(d["mean_ttft_ms"], 2),
            "tpot_ms": round(d["mean_tpot_ms"], 2),
            "accept_length": round(d.get("accept_length", 0) or 0, 2),
        }
    )
write_csv("limits.csv", limits)

telemetry = []
for path in sorted(RAW.glob("*.telemetry.csv")):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            try:
                number = lambda key: float(re.sub(r"[^0-9.]", "", row[key]))
                rows.append(
                    (
                        number("utilization.gpu [%]"),
                        number("power.draw [W]"),
                        number("memory.used [MiB]"),
                        number("temperature.gpu"),
                    )
                )
            except (KeyError, ValueError):
                pass
    active = [row for row in rows if row[0] >= 10]
    sample = active or rows
    telemetry.append(
        {
            "test": path.name.removesuffix(".telemetry.csv"),
            "samples": len(rows),
            "active_samples": len(active),
            "mean_active_gpu_util_pct": round(sum(x[0] for x in sample) / len(sample), 1),
            "max_gpu_util_pct": round(max(x[0] for x in sample), 1),
            "mean_active_power_w": round(sum(x[1] for x in sample) / len(sample), 1),
            "max_power_w": round(max(x[1] for x in sample), 1),
            "max_vram_mib": round(max(x[2] for x in sample)),
            "max_temp_c": round(max(x[3] for x in sample)),
        }
    )
write_csv("telemetry.csv", telemetry)

# Decode comparison, grouped by workload.
fig, ax = plt.subplots(figsize=(8, 4.8))
labels = list(configs)
x = range(len(labels))
micro = [next(r["output_tok_s"] for r in summary if r["config"] == label and r["workload"] == "micro") for label in labels]
official = [next(r["output_tok_s"] for r in summary if r["config"] == label and r["workload"] == "official") for label in labels]
width = 0.36
ax.bar([i - width / 2 for i in x], micro, width, label="1K input / 256 output")
ax.bar([i + width / 2 for i in x], official, width, label="8K input / 1K output")
ax.set_xticks(list(x), labels)
ax.set_ylabel("Output tokens/s")
ax.set_title("Qwen3.8-27B NVFP4 — RTX 5090 single-stream decode")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(PLOTS / f"decode-comparison.{ext}", dpi=180)
plt.close(fig)

# Concurrency throughput and TTFT.
fig, ax1 = plt.subplots(figsize=(8, 4.8))
cs = [r["requested_concurrency"] for r in concurrency]
tps = [r["output_tok_s"] for r in concurrency]
ttft = [r["mean_ttft_ms"] for r in concurrency]
ax1.plot(cs, tps, "o-", color="#2563eb", linewidth=2, label="Aggregate output tok/s")
ax1.set_xlabel("Requested concurrency")
ax1.set_ylabel("Aggregate output tokens/s", color="#2563eb")
ax1.tick_params(axis="y", labelcolor="#2563eb")
ax1.set_xticks(cs)
ax1.grid(alpha=0.25)
ax2 = ax1.twinx()
ax2.plot(cs, ttft, "s--", color="#dc2626", linewidth=2, label="Mean TTFT")
ax2.set_ylabel("Mean TTFT (ms)", color="#dc2626")
ax2.tick_params(axis="y", labelcolor="#dc2626")
ax1.set_title("MTP-5 throughput scaling — 1K input / 256 output")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(PLOTS / f"concurrency.{ext}", dpi=180)
plt.close(fig)

print(f"Wrote summaries to {HERE}")
