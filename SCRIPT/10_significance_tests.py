"""
Statistical robustness tests for the RAG / LoRA / Combined comparison.
No rerun / GPU needed: operates on saved per-pair predictions, using
per-project CLEAN F2 (parse failures excluded).

For each of 5 comparisons (A vs B) over the 8 projects:
  - mean and median per-project F2 difference (A - B)
  - sign test  (how many of 8 projects A wins) + two-sided binomial p
  - paired Wilcoxon signed-rank p
  - bootstrap 95% CI of the mean difference (resampling the 8 projects)

Interpretation note: with only 8 projects, effect sizes and win counts are
more informative than p-values; these are reported as robustness checks.
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon, binomtest

RESULTS = "RESULTS"
PROJ = ["AAH","BEAM","CB","FH","JBIDE","KEYCLOAK","KOGITO","PROJQUAY"]

METHOD_PATHS = {
    "ZeroShot": "ZERO_SHOT_QWEN_HARD/{p}_predictions.json",
    "RAG-B":    "RAG_STAGE1_V3_8192/RAG_B/PREDICTIONS/{p}_predictions.json",
    "LoRA-V4":  "LORA_RERUN_V3/V4_EFFICIENCY/PREDICTIONS/{p}_predictions.json",
    "Combined": "COMBINED_RERUN_V3/V4_EFFICIENCY_RAG_B_8192/PREDICTIONS/{p}_predictions.json",
}

def clean_f2(path):
    """Per-project CLEAN F2: drop pairs whose prediction is None (parse failure)."""
    recs = json.load(open(path))
    tp=fp=fn=0
    for r in recs:
        pred = r.get("prediction")
        if pred is None:      # parse failure -> excluded from clean metric
            continue
        lab = int(r["label"]); pred = int(pred)
        if pred==1 and lab==1: tp+=1
        elif pred==1 and lab==0: fp+=1
        elif pred==0 and lab==1: fn+=1
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r_ = tp/(tp+fn) if (tp+fn) else 0.0
    return (5*p*r_/(4*p+r_)) if (4*p+r_) else 0.0

# per-project clean F2 for each method
f2 = {m: np.array([clean_f2(os.path.join(RESULTS, pat.format(p=p))) for p in PROJ])
      for m, pat in METHOD_PATHS.items()}

print("Per-project CLEAN F2:")
print(f"  {'Project':<10}" + "".join(f"{m:>11}" for m in METHOD_PATHS))
for i,p in enumerate(PROJ):
    print(f"  {p:<10}" + "".join(f"{f2[m][i]:>11.4f}" for m in METHOD_PATHS))
print(f"  {'MACRO':<10}" + "".join(f"{f2[m].mean():>11.4f}" for m in METHOD_PATHS))

COMPARISONS = [
    ("RAG-B","LoRA-V4"),       # RQ2.1 main: best RAG vs best LoRA
    ("Combined","RAG-B"),      # does LoRA add value beyond best RAG?
    ("Combined","LoRA-V4"),    # does RAG add value beyond best LoRA?
    ("RAG-B","ZeroShot"),      # adaptation (RAG) vs zero-shot
    ("LoRA-V4","ZeroShot"),    # adaptation (LoRA) vs zero-shot
]

rng = np.random.default_rng(42)
B = 10000
rows = []
print("\nPaired comparisons over 8 projects (clean F2, delta = A - B):")
hdr = f"  {'A vs B':<22}{'mean d':>9}{'median d':>10}{'wins A':>8}{'sign p':>9}{'Wilcox p':>10}{'boot 95% CI':>22}"
print(hdr); print("  "+"-"*len(hdr))
for a,b in COMPARISONS:
    d = f2[a]-f2[b]
    mean_d=float(np.mean(d)); med_d=float(np.median(d))
    wins=int(np.sum(d>0)); losses=int(np.sum(d<0)); ties=int(np.sum(d==0))
    ntb=wins+losses
    signp = binomtest(wins, ntb, 0.5).pvalue if ntb>0 else float('nan')
    try: wp = wilcoxon(d).pvalue
    except Exception: wp=float('nan')
    boot=np.array([np.mean(d[rng.integers(0,8,8)]) for _ in range(B)])
    lo,hi=np.percentile(boot,[2.5,97.5])
    rows.append(dict(comparison=f"{a} vs {b}", mean_diff=round(mean_d,4), median_diff=round(med_d,4),
                     wins_A=wins, losses_A=losses, ties=ties, sign_test_p=round(float(signp),4),
                     wilcoxon_p=round(float(wp),4), boot_ci_low=round(float(lo),4), boot_ci_high=round(float(hi),4)))
    print(f"  {a+' vs '+b:<22}{mean_d:>+9.4f}{med_d:>+10.4f}{str(wins)+'/'+str(ntb):>8}{signp:>9.4f}{wp:>10.4f}   [{lo:+.4f}, {hi:+.4f}]")

out = {"metric":"clean macro F2","n_projects":8,
       "per_project_f2":{m:f2[m].round(4).tolist() for m in METHOD_PATHS},
       "macro_f2":{m:round(float(f2[m].mean()),4) for m in METHOD_PATHS},
       "comparisons":rows,
       "note":"With 8 projects, effect sizes and win counts are more informative than p-values; reported as robustness checks."}
json.dump(out, open(os.path.join(RESULTS,"significance_tests_clean_f2.json"),"w"), indent=2)
print("\nSaved -> RESULTS/significance_tests_clean_f2.json")
