"""
Report Generator
================
Reads all results and produces a comprehensive analysis with:
- Best params per mode
- Parameter sensitivity analysis
- Stability report
- Final recommendations
"""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results")


def load_results(pattern: str) -> list[dict]:
    """Load all results matching a filename pattern."""
    results = []
    for f in RESULTS_DIR.glob(pattern):
        with open(f) as fh:
            for line in fh:
                try:
                    results.append(json.loads(line))
                except:
                    pass
    return results


def sensitivity_analysis(results: list[dict], param_name: str) -> dict:
    """Analyze how a single parameter affects scores, holding others constant."""
    by_value = defaultdict(list)
    for r in results:
        if "error" not in r:
            val = r["params"][param_name]
            by_value[val].append(r["grade"]["weighted_score"])

    analysis = {}
    for val, scores in sorted(by_value.items()):
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        analysis[val] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "n": len(scores),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }
    return analysis


def generate_report(mode: str):
    """Generate a full report for a mode."""
    # Load all phases
    coarse = load_results(f"coarse_{mode}.jsonl")
    focused = load_results(f"focused_{mode}.jsonl")
    fine = load_results(f"fine_{mode}.jsonl")

    all_results = coarse + focused + fine
    valid = [r for r in all_results if "error" not in r]

    if not valid:
        print(f"No valid results for {mode}")
        return

    print(f"\n{'='*70}")
    print(f"  REPORT: {mode.upper()} MODE")
    print(f"  Total results: {len(all_results)} ({len(valid)} valid, {len(all_results)-len(valid)} errors)")
    print(f"{'='*70}")

    # ── Overall best combos ──
    by_params = defaultdict(list)
    for r in valid:
        key = json.dumps(r["params"], sort_keys=True)
        by_params[key].append(r)

    combo_scores = []
    for key, runs in by_params.items():
        scores = [r["grade"]["weighted_score"] for r in runs]
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        combo_scores.append({
            "params": json.loads(key),
            "mean": mean,
            "std": std,
            "min": min(scores),
            "max": max(scores),
            "combined": mean - 0.5 * std + 0.1 * min(scores),
            "n": len(runs),
        })

    combo_scores.sort(key=lambda x: x["combined"], reverse=True)

    print(f"\n  TOP 5 PARAMETER COMBINATIONS:")
    print(f"  {'Rank':<5} {'Combined':<10} {'Mean±Std':<16} {'Min-Max':<14} {'N':<5} {'Parameters'}")
    print(f"  {'-'*90}")
    for i, cs in enumerate(combo_scores[:5]):
        p = cs["params"]
        param_str = f"T={p['temperature']:.2f}  top_p={p['top_p']:.2f}  top_k={p['top_k']:<3}  min_p={p['min_p']:.3f}  rep={p['repeat_penalty']:.2f}"
        print(f"  {i+1:<5} {cs['combined']:<10.4f} {cs['mean']:.3f}±{cs['std']:.3f}    {cs['min']:.3f}-{cs['max']:.3f}  {cs['n']:<5} {param_str}")

    # ── Parameter sensitivity ──
    print(f"\n  PARAMETER SENSITIVITY ANALYSIS:")
    for param in ["temperature", "top_p", "top_k", "min_p", "repeat_penalty"]:
        sa = sensitivity_analysis(valid, param)
        print(f"\n  {param}:")
        print(f"    {'Value':<10} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'N'}")
        for val, stats in sa.items():
            print(f"    {val:<10} {stats['mean']:<8.4f} {stats['std']:<8.4f} {stats['min']:<8.4f} {stats['max']:<8.4f} {stats['n']}")

    # ── Per-prompt breakdown for best combo ──
    if combo_scores:
        best = combo_scores[0]
        best_key = json.dumps(best["params"], sort_keys=True)
        best_runs = by_params[best_key]

        print(f"\n  PER-PROMPT BREAKDOWN (Best combo: T={best['params']['temperature']}, top_p={best['params']['top_p']}):")
        by_prompt = defaultdict(list)
        for r in best_runs:
            by_prompt[r["prompt_id"]].append(r)

        for pid, runs in sorted(by_prompt.items()):
            scores = [r["grade"]["weighted_score"] for r in runs]
            dims = defaultdict(list)
            for r in runs:
                for d, v in r["grade"]["dimensions"].items():
                    dims[d].append(v)

            mean = sum(scores) / len(scores)
            dim_str = " | ".join(f"{d}={sum(vs)/len(vs):.2f}" for d, vs in sorted(dims.items()))
            print(f"    {pid:<20} score={mean:.3f} (n={len(runs)})  [{dim_str}]")

    # ── Speed analysis ──
    print(f"\n  SPEED ANALYSIS:")
    elapsed_list = [r["elapsed"] for r in valid if r.get("elapsed")]
    if elapsed_list:
        avg_time = sum(elapsed_list) / len(elapsed_list)
        print(f"    Average response time: {avg_time:.1f}s")
        print(f"    Min/Max: {min(elapsed_list):.1f}s / {max(elapsed_list):.1f}s")

    # ── Derail analysis ──
    print(f"\n  DERAIL ANALYSIS (scores < 0.3):")
    derailed = [r for r in valid if r["grade"]["weighted_score"] < 0.3]
    print(f"    Total derailed: {len(derailed)} / {len(valid)} ({100*len(derailed)/len(valid):.1f}%)")
    if derailed:
        derail_by_temp = defaultdict(int)
        for r in derailed:
            derail_by_temp[r["params"]["temperature"]] += 1
        print(f"    Derails by temperature: {dict(sorted(derail_by_temp.items()))}")

    # ── Flag analysis ──
    print(f"\n  FLAG ANALYSIS:")
    all_flags = defaultdict(int)
    for r in valid:
        for f in r["grade"].get("flags", []):
            all_flags[f] += 1
    if all_flags:
        for flag, count in sorted(all_flags.items(), key=lambda x: -x[1]):
            print(f"    {flag}: {count} ({100*count/len(valid):.1f}%)")
    else:
        print(f"    No flags raised")

    # ── Save full report as JSON ──
    report = {
        "mode": mode,
        "total_results": len(all_results),
        "valid_results": len(valid),
        "top_5_combos": [
            {
                "rank": i + 1,
                "params": cs["params"],
                "combined": round(cs["combined"], 4),
                "mean": round(cs["mean"], 4),
                "std": round(cs["std"], 4),
                "min": round(cs["min"], 4),
                "max": round(cs["max"], 4),
                "n": cs["n"],
            }
            for i, cs in enumerate(combo_scores[:5])
        ],
        "best_params": combo_scores[0]["params"] if combo_scores else None,
    }
    report_file = RESULTS_DIR / f"report_{mode}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_file}")

    return report


if __name__ == "__main__":
    import sys
    modes = sys.argv[1:] if len(sys.argv) > 1 else ["planner", "coder"]
    reports = {}
    for m in modes:
        reports[m] = generate_report(m)

    # ── Final comparison ──
    if len(reports) == 2 and all(reports.values()):
        print(f"\n{'='*70}")
        print(f"  FINAL COMPARISON: PLANNER vs CODER OPTIMAL SETTINGS")
        print(f"{'='*70}")
        for mode, r in reports.items():
            if r and r.get("best_params"):
                p = r["best_params"]
                print(f"\n  {mode.upper()}:")
                print(f"    temperature:    {p['temperature']}")
                print(f"    top_p:          {p['top_p']}")
                print(f"    top_k:          {p['top_k']}")
                print(f"    min_p:          {p['min_p']}")
                print(f"    repeat_penalty: {p['repeat_penalty']}")
