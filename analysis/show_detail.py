import json, sys
from pathlib import Path

target = sys.argv[1] if len(sys.argv) > 1 else "gpt-oss-20b"
rows = json.loads(Path("analysis/forensics.json").read_text())
r = next(x for x in rows if x["model"] == target)

print(f"=== {r['model']} | threshold {r['threshold']:,} ===")
print(f"MRF {r['mrf']['value']:+.4f}  ci95 {r['mrf']['ci95']}  perm_p {r['mrf']['perm_p']:.3f}")
g = r["p_gap"]
print(f"\nP(final > threshold) gap: {g['value']:+.3f}  ci95 "
      f"[{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]")
print(f"  ({g['definition']})")

print("\nper condition:")
for c, d in r["conditions"].items():
    print(f"  {c:<11} P(>thr)={d.get('p_above_threshold'):.3f} "
          f"ci95=[{d['p_above_ci95'][0]:.3f},{d['p_above_ci95'][1]:.3f}]  "
          f"median_final={d.get('median_final'):,.0f}  "
          f"within5%ofthr={d.get('frac_within_5pct_of_threshold')}  "
          f"exactly_at_thr={d.get('frac_exactly_at_threshold')}")
    print(f"              median_rel_dev={d.get('median_rel_deviation'):+.3f}  "
          f"cot_chars={d.get('median_cot_chars'):,.0f}  "
          f"n_est={d.get('median_n_estimates')}  "
          f"report_shift_median={d['report_shift'].get('median')}")

print("\nstart-side strata:")
for k, v in r["start_side"].items():
    fav = "favoured" if v["already_favoured_at_start"] else "unfavoured"
    md = v["median_drift"]
    print(f"  {k:<26} n={v['n']:>3} {fav:<10} median_drift="
          f"{md:+.4f}" if md is not None else f"  {k}: n=0")
