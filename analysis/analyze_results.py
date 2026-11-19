"""Analyze k6 benchmark results and compute Azure cost & break-even point."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Azure East Asia pricing reference (USD, pay-as-you-go; update from Azure Pricing Calculator on test date)
ACA_VCPU_PER_SEC = 0.000024  # USD per vCPU-second (active)
ACA_GIB_PER_SEC = 0.000003   # USD per GiB-second (active)
ACA_REQ_PER_MILLION = 0.40   # USD per 1M requests
ACA_FREE_VCPU_S = 180_000    # Monthly free grant: 180,000 vCPU-s
ACA_FREE_GIB_S = 360_000     # Monthly free grant: 360,000 GiB-s
ACA_FREE_REQS = 2_000_000    # Monthly free grant: 2M requests

VM_B2S_HOURLY = 0.0416       # Standard_B2s (2 vCPU, 4 GiB) Linux East Asia ~ $30.37/mo
VM_DISK_MONTHLY = 2.40       # 32 GiB Standard SSD
VM_IP_MONTHLY = 3.65         # Standard Static Public IP
HOURS_PER_MONTH = 730


def parse_k6_summary(path: Path) -> dict | None:
    m = re.match(r"^(aca|vm|local)_b(\d+)_vu(\d+)_r(\d+)\.json$", path.name)
    if not m:
        return None
    env_name, batch, vu, rep = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})

    dur = metrics.get("http_req_duration", {})
    reqs = metrics.get("http_reqs", {})
    failed = metrics.get("http_req_failed", {})
    srv = metrics.get("server_latency_ms", {})

    return {
        "env": env_name,
        "batch_size": batch,
        "vus": vu,
        "repeat": rep,
        "p50_ms": float(dur.get("med", dur.get("p(50)", 0.0))),
        "p95_ms": float(dur.get("p(95)", 0.0)),
        "p99_ms": float(dur.get("p(99)", dur.get("max", 0.0))),
        "server_p50_ms": float(srv.get("med", 0.0)),
        "server_p95_ms": float(srv.get("p(95)", 0.0)),
        "rps": float(reqs.get("rate", 0.0)),
        "total_reqs": int(reqs.get("count", 0)),
        "error_rate": float(failed.get("value", 0.0)),
    }


def aca_monthly_cost(
    n_requests: float,
    bill_time_s: float,
    vcpus: float = 1.0,
    mem_gib: float = 2.0,
    effective_concurrency: float = 1.0,
    include_free_grant: bool = True,
) -> float:
    total_vcpu_s = n_requests * (bill_time_s * vcpus / effective_concurrency)
    total_gib_s = n_requests * (bill_time_s * mem_gib / effective_concurrency)

    if include_free_grant:
        bill_vcpu_s = max(0.0, total_vcpu_s - ACA_FREE_VCPU_S)
        bill_gib_s = max(0.0, total_gib_s - ACA_FREE_GIB_S)
        bill_reqs = max(0.0, n_requests - ACA_FREE_REQS)
    else:
        bill_vcpu_s, bill_gib_s, bill_reqs = total_vcpu_s, total_gib_s, n_requests

    return (
        bill_vcpu_s * ACA_VCPU_PER_SEC
        + bill_gib_s * ACA_GIB_PER_SEC
        + (bill_reqs / 1_000_000.0) * ACA_REQ_PER_MILLION
    )


def vm_monthly_cost() -> float:
    return HOURS_PER_MONTH * VM_B2S_HOURLY + VM_DISK_MONTHLY + VM_IP_MONTHLY


def plot_cost_breakeven(bill_time_s: float, effective_concurrency: float, out_dir: Path) -> pd.DataFrame:
    ns = np.logspace(3, 7.5, 200)
    cost_aca_paid = [
        aca_monthly_cost(n, bill_time_s, effective_concurrency=effective_concurrency, include_free_grant=False)
        for n in ns
    ]
    cost_aca_free = [
        aca_monthly_cost(n, bill_time_s, effective_concurrency=effective_concurrency, include_free_grant=True)
        for n in ns
    ]
    cost_vm = vm_monthly_cost()

    # Exact break-even without free grant
    unit_cost = (bill_time_s * (1.0 * ACA_VCPU_PER_SEC + 2.0 * ACA_GIB_PER_SEC) / effective_concurrency) + (
        ACA_REQ_PER_MILLION / 1_000_000.0
    )
    breakeven_no_grant = cost_vm / unit_cost if unit_cost > 0 else float("inf")

    plt.figure(figsize=(8, 5))
    plt.plot(ns, cost_aca_paid, label="Azure Container Apps (no free grant)", linewidth=2)
    plt.plot(ns, cost_aca_free, label="Azure Container Apps (with free grant)", linestyle="--", linewidth=2)
    plt.axhline(cost_vm, color="tab:red", label=f"Azure VM B2s (${cost_vm:.2f}/mo)", linewidth=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Monthly Inference Requests (N)")
    plt.ylabel("Estimated Monthly Cost (USD)")
    plt.title(f"Cost & Break-Even Analysis (t={bill_time_s*1000:.0f}ms, c_eff={effective_concurrency})")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cost_breakeven.png", dpi=150)
    plt.close()

    rows = []
    for n in [1_000, 100_000, 1_000_000, 10_000_000]:
        rows.append(
            {
                "monthly_requests": n,
                "aca_usd_no_grant": round(
                    aca_monthly_cost(n, bill_time_s, effective_concurrency=effective_concurrency, include_free_grant=False),
                    4,
                ),
                "aca_usd_with_grant": round(
                    aca_monthly_cost(n, bill_time_s, effective_concurrency=effective_concurrency, include_free_grant=True),
                    4,
                ),
                "vm_b2s_usd": round(cost_vm, 2),
                "breakeven_reqs_no_grant": int(breakeven_no_grant),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--default-latency-s", type=float, default=0.045)
    parser.add_argument("--effective-concurrency", type=float, default=4.0)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    proc_dir = Path(args.out_dir) / "processed"
    charts_dir = Path(args.out_dir) / "charts"
    proc_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    records = []
    if raw_dir.exists():
        for p in sorted(raw_dir.glob("*.json")):
            rec = parse_k6_summary(p)
            if rec:
                records.append(rec)

    bill_time_s = args.default_latency_s
    if records:
        df = pd.DataFrame(records)
        summary = (
            df.groupby(["env", "batch_size", "vus"], as_index=False)
            .agg(
                p50_ms_mean=("p50_ms", "mean"),
                p50_ms_std=("p50_ms", "std"),
                p95_ms_mean=("p95_ms", "mean"),
                p95_ms_std=("p95_ms", "std"),
                rps_mean=("rps", "mean"),
                rps_std=("rps", "std"),
                error_rate_mean=("error_rate", "mean"),
            )
            .round(3)
        )
        summary.to_csv(proc_dir / "summary_table.csv", index=False)
        print(summary.to_string(index=False))

        # Use measured batch=1 p50 server latency if available
        b1 = df[df["batch_size"] == 1]
        if not b1.empty and b1["server_p50_ms"].mean() > 0:
            bill_time_s = float(b1["server_p50_ms"].mean() / 1000.0)

        # Plot p95 latency vs batch size
        plt.figure(figsize=(8, 5))
        for (env_name, vu), grp in summary.groupby(["env", "vus"]):
            plt.errorbar(
                grp["batch_size"],
                grp["p95_ms_mean"],
                yerr=grp["p95_ms_std"].fillna(0),
                marker="o",
                capsize=4,
                label=f"{env_name} (VUs={vu})",
            )
        plt.xlabel("Batch Size")
        plt.ylabel("p95 Latency (ms)")
        plt.title("p95 Latency vs Batch Size & Concurrency")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(charts_dir / "latency_vs_batch.png", dpi=150)
        plt.close()
    else:
        print("No raw k6 JSON files found in results/raw/. Generating cost/break-even model only.")

    cost_df = plot_cost_breakeven(bill_time_s, args.effective_concurrency, charts_dir)
    cost_df.to_csv(proc_dir / "cost_comparison.csv", index=False)
    print("\nCost & Break-Even Table:")
    print(cost_df.to_string(index=False))


if __name__ == "__main__":
    main()

