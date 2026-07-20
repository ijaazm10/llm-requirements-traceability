"""
Thesis Figure Generator
=======================
Generates all publication-quality figures for the thesis from existing result JSON files.
Run this locally on Windows — reads from RESULTS/, saves to RESULTS/FIGURES/.

Usage:
    python 11_generate_thesis_figures.py

Figures produced:
    Fig 1  — Main comparison: all methods, Macro F2 horizontal bar
    Fig 2  — Per-project F2 heatmap (4 key methods)
    Fig 3  — RAG ablation: A/B/C/D, clean + conservative F2
    Fig 4  — LoRA ablation: V1–V6, clean + conservative F2
    Fig 5  — Combined system breakdown (LoRA alone / RAG alone / Combined)
    Fig 6  — Statistical significance forest plot (bootstrap 95% CI)
    Fig 7  — Similarity inversion: pos_sim vs neg_sim per project (test split)
    Fig 8  — Precision–Recall trade-off scatter for all LLM methods
    Fig 9  — Parse failure / reliability chart (RQ3)

Author: Thesis Work
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
})

RESULTS = Path("RESULTS")
FIGURES = RESULTS / "FIGURES"
FIGURES.mkdir(exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "baseline":  "#9E9E9E",
    "zeroshot":  "#5C85D6",
    "rag":       "#26A69A",
    "rag_light": "#80CBC4",
    "lora":      "#EF6C00",
    "lora_light":"#FFCC80",
    "combined":  "#1B5E20",
    "cloud":     "#7B1FA2",
    "sig":       "#C62828",
    "neutral":   "#455A64",
}

PROJECTS = ["AAH","BEAM","CB","FH","JBIDE","KEYCLOAK","KOGITO","PROJQUAY"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — MAIN COMPARISON  (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_main_comparison():
    # ---- hard-coded from your verified result JSONs ----
    methods = [
        # (label,               F2_clean, F2_cons,  colour,         group)
        ("VSM (TF-IDF)",        0.4460,   None,     C["baseline"],  "Classical\nBaselines"),
        ("Frozen BERT",         0.5882,   None,     C["baseline"],  "Classical\nBaselines"),
        ("SBERT (MPNet)",       0.6045,   None,     C["baseline"],  "Classical\nBaselines"),
        ("Zero-Shot Gemma 4 31B",0.6938,  0.6912,   C["zeroshot"],  "Zero-Shot\nLLM"),
        ("GPT-5.4",             0.6991,   0.6991,   C["cloud"],     "Cloud\nZero-Shot"),
        ("Claude Sonnet 4-6",   0.7511,   0.7511,   C["cloud"],     "Cloud\nZero-Shot"),
        ("LoRA V4 (champion)",  0.7537,   0.7522,   C["lora"],      "LoRA\nFine-Tuning"),
        ("RAG-D Hybrid 2+2",    0.7664,   0.7573,   C["rag_light"], "RAG\nAugmentation"),
        ("RAG-C Qwen3 4+0",     0.7751,   0.7546,   C["rag_light"], "RAG\nAugmentation"),
        ("RAG-A MPNet 2+2",     0.7846,   0.7796,   C["rag"],       "RAG\nAugmentation"),
        ("RAG-B Qwen3 2+2",     0.7878,   0.7800,   C["rag"],       "RAG\nAugmentation"),
        ("Combined V4 + RAG-B", 0.7947,   0.7868,   C["combined"],  "Combined\nSystem"),
    ]

    labels   = [m[0] for m in methods]
    f2_clean = [m[1] for m in methods]
    f2_cons  = [m[2] for m in methods]
    colours  = [m[3] for m in methods]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(labels))

    bars = ax.barh(y, f2_clean, height=0.55, color=colours, zorder=3)

    # conservative F2 as thin darker overlay bar
    for i, fc in enumerate(f2_cons):
        if fc is not None:
            ax.barh(y[i], fc, height=0.55, color="none",
                    edgecolor="black", linewidth=0.8, linestyle=":", zorder=4)

    # value labels
    for i, (bar, v) in enumerate(zip(bars, f2_clean)):
        ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=8.5, color="#222222")

    # group dividers
    group_breaks = [2.5, 3.5, 5.5, 6.5, 10.5]
    for gb in group_breaks:
        ax.axhline(gb, color="#CCCCCC", linewidth=0.8, zorder=2)

    # group labels on right spine
    group_info = [
        (1,    "Classical Baselines"),
        (3,    "Zero-Shot LLM"),
        (4.5,  "Cloud Zero-Shot"),
        (6,    "LoRA Fine-Tuning"),
        (8.75, "RAG Augmentation"),
        (11,   "Combined"),
    ]
    for ypos, glabel in group_info:
        ax.text(1.003, ypos / (len(labels)-1+0.5),
                glabel, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.5, color="#555555",
                style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.38, 0.86)
    ax.set_xlabel("Macro F2 (clean, 8-project average)")
    ax.set_title("Fig 1 — Overall Performance Comparison: All Methods", fontweight="bold", pad=10)

    # legend for conservative overlay
    cons_patch = mpatches.Patch(facecolor="none", edgecolor="black",
                                linestyle=":", linewidth=0.8,
                                label="Conservative F2 (parse failures → FN)")
    ax.legend(handles=[cons_patch], loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out = FIGURES / "Fig1_main_comparison.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf",".png"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — PER-PROJECT HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def fig2_heatmap():
    sig_data = load(RESULTS / "significance_tests_clean_f2.json")
    pf = sig_data["per_project_f2"]

    # Add Claude and GPT per-project from their result files
    claude_res = load(RESULTS / "CLAUDE_ZERO_SHOT_BATCH_V3" / "claude-sonnet-4-6_matched_single_user_prompt_v1_batch" / "results.json")
    gpt_res    = load(RESULTS / "OPENAI_ZERO_SHOT_BATCH_V3" / "gpt-5_4_merged_matched_single_user_prompt_v1_batch" / "results.json")

    claude_f2 = {r["project"]: r["metrics_clean"]["f2"] for r in claude_res["per_project"]}
    gpt_f2    = {r["project"]: r["metrics_clean"]["f2"] for r in gpt_res["per_project"]}

    methods_ordered = [
        ("Zero-Shot Gemma",  [pf["ZeroShot"][i]   for i in range(8)]),
        ("GPT-5.4",          [gpt_f2[p]           for p in PROJECTS]),
        ("Claude Sonnet",    [claude_f2[p]         for p in PROJECTS]),
        ("LoRA V4",          [pf["LoRA-V4"][i]    for i in range(8)]),
        ("RAG-B",            [pf["RAG-B"][i]      for i in range(8)]),
        ("Combined",         [pf["Combined"][i]   for i in range(8)]),
    ]

    labels_m = [m[0] for m in methods_ordered]
    data = np.array([m[1] for m in methods_ordered])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0.45, vmax=0.98)

    ax.set_xticks(range(8))
    ax.set_xticklabels(PROJECTS, rotation=30, ha="right")
    ax.set_yticks(range(len(labels_m)))
    ax.set_yticklabels(labels_m)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            colour = "white" if v > 0.80 else "#333333"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color=colour, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Clean F2", shrink=0.85)
    ax.set_title("Fig 2 — Per-Project F2 Heatmap (Key Methods)", fontweight="bold", pad=10)
    plt.tight_layout()
    out = FIGURES / "Fig2_per_project_heatmap.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf",".png"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — RAG ABLATION
# ─────────────────────────────────────────────────────────────────────────────
def fig3_rag_ablation():
    rag_data = load(RESULTS / "RAG_STAGE2_HYBRID_V3_8192" / "MASTER_RAG_ABCD_COMPARISON.json")

    configs   = ["RAG_A", "RAG_B", "RAG_C", "RAG_D"]
    labels    = ["RAG-A\nMPNet 2+2", "RAG-B\nQwen3 2+2", "RAG-C\nQwen3 4+0", "RAG-D\nHybrid 2+2"]
    f2_clean  = [rag_data["configs"][c]["macro_clean"]["f2"]        for c in configs]
    f2_cons   = [rag_data["configs"][c]["macro_conservative"]["f2"] for c in configs]
    fail_rate = [rag_data["configs"][c]["deployment_rollup"]["macro_format_failure_rate_pct"] for c in configs]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w/2, f2_clean, w, label="Clean F2",        color=C["rag"],       zorder=3)
    bars2 = ax1.bar(x + w/2, f2_cons,  w, label="Conservative F2", color=C["rag_light"], zorder=3, hatch="//")
    ax2.plot(x, fail_rate, "o--", color=C["sig"], linewidth=1.5, markersize=6,
             label="Parse fail rate (%)", zorder=4)

    for bar, v in zip(bars1, f2_clean):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.002, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.70, 0.82)
    ax1.set_ylabel("Macro F2")
    ax2.set_ylabel("Parse Failure Rate (%)", color=C["sig"])
    ax2.tick_params(axis="y", colors=C["sig"])
    ax2.set_ylim(0, 6)

    lines1, ll1 = ax1.get_legend_handles_labels()
    lines2, ll2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, ll1+ll2, loc="lower right", framealpha=0.9)

    ax1.set_title("Fig 3 — RAG Ablation Study (A–D)", fontweight="bold", pad=10)
    plt.tight_layout()
    out = FIGURES / "Fig3_rag_ablation.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf",".png"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — LoRA ABLATION
# ─────────────────────────────────────────────────────────────────────────────
def fig4_lora_ablation():
    lora_data = load(RESULTS / "LORA_RERUN_V3" / "MASTER_COMPARISON.json")

    versions = ["V1_NAIVE","V2_BALANCED","V3_STABILIZED","V4_EFFICIENCY","V5_SYNTHESIS","V6_MLP"]
    x_labels = ["V1\nNaive\n(no upsample)","V2\nBalanced\n(baseline)","V3\nStabilized\n(LR halved)",
                 "V4\nEfficiency\n(R halved)★","V5\nSynthesis\n(R+LR halved)","V6\nMLP\n(+MLP modules)"]

    f2_clean = [lora_data["versions"][v]["macro_clean"]["f2"] for v in versions]
    f2_cons  = [lora_data["versions"][v]["macro_conservative"]["f2"] for v in versions]

    colours_bar = [C["lora_light"] if v != "V4_EFFICIENCY" else C["lora"] for v in versions]

    x = np.arange(len(versions))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(x - w/2, f2_clean, w, color=colours_bar, label="Clean F2", zorder=3)
    ax.bar(x + w/2, f2_cons,  w, color=colours_bar, label="Conservative F2",
           zorder=3, alpha=0.55, hatch="//")

    for bar, v in zip(bars, f2_clean):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.001, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)

    # zero-shot reference line
    ax.axhline(0.6938, color=C["zeroshot"], linestyle="--", linewidth=1.4,
               label="Zero-Shot Gemma 4 31B (F2=0.6938)")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8.5)
    ax.set_ylim(0.58, 0.78)
    ax.set_ylabel("Macro F2")
    ax.set_title("Fig 4 — LoRA Fine-Tuning Ablation (V1–V6)", fontweight="bold", pad=10)

    lora_patch = mpatches.Patch(color=C["lora"], label="V4 Champion")
    lora_l_patch = mpatches.Patch(color=C["lora_light"], label="Other versions")
    hatch_patch = mpatches.Patch(facecolor="none", edgecolor="#555", hatch="//", label="Conservative F2")
    ax.legend(handles=[lora_patch, lora_l_patch, hatch_patch,
                        plt.Line2D([0],[0], color=C["zeroshot"], linestyle="--", linewidth=1.4,
                                   label="Zero-Shot Gemma baseline")],
              loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out = FIGURES / "Fig4_lora_ablation.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf",".png"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — COMBINED SYSTEM BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
def fig5_combined():
    comb_data = load(RESULTS / "COMBINED_RERUN_V3" / "MASTER_COMBINED_COMPARISON.json")
    cfgs = comb_data["configurations"]

    methods = ["LoRA V4\nalone", "RAG-B\nalone", "Combined\nV4 + RAG-B"]
    keys    = ["LORA_V4_ALONE","RAG_B_ALONE","V4_EFFICIENCY_RAG_B_8192"]
    cols    = [C["lora"], C["rag"], C["combined"]]

    metrics_labels = ["Precision", "Recall", "F1", "F2", "Accuracy"]
    metric_keys    = ["precision", "recall", "f1", "f2", "accuracy"]

    data = []
    for k in keys:
        m = cfgs[k]["macro_clean"]
        data.append([m[mk] for mk in metric_keys])
    data = np.array(data)

    x = np.arange(len(metrics_labels))
    w = 0.22
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (label, colour) in enumerate(zip(methods, cols)):
        bars = ax.bar(x + (i-1)*w, data[i], w, color=colour, label=label, zorder=3)
        for bar, v in zip(bars, data[i]):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)
    ax.set_ylim(0.60, 0.93)
    ax.set_ylabel("Score")
    ax.set_title("Fig 5 — Combined System vs Components (Macro Metrics)", fontweight="bold", pad=10)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    out = FIGURES / "Fig5_combined_breakdown.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf",".png"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — STATISTICAL SIGNIFICANCE FOREST PLOT
# ─────────────────────────────────────────────────────────────────────────────
def fig6_significance():
    sig = load(RESULTS / "significance_tests_clean_f2.json")
    comps = sig["comparisons"]

    labels   = [c["comparison"]   for c in comps]
    means    = [c["mean_diff"]    for c in comps]
    lo       = [c["boot_ci_low"]  for c in comps]
    hi       = [c["boot_ci_high"] for c in comps]
    wilcox_p = [c["wilcoxon_p"]   for c in comps]
    wins     = [c["wins_A"]       for c in comps]

    y = np.arange(len(labels))[::-1]  # reverse so first comparison is at top

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, (yi, m, l, h, wp, w_) in enumerate(zip(y, means, lo, hi, wilcox_p, wins)):
        ci_crosses_zero = l <= 0 <= h
        colour = C["neutral"] if ci_crosses_zero else C["sig"]
        marker = "o" if not ci_crosses_zero else "D"
        ms = 8

        # CI line
        ax.plot([l, h], [yi, yi], color=colour, linewidth=2.0, zorder=3)
        # mean dot
        ax.plot(m, yi, marker=marker, color=colour, markersize=ms, zorder=4)
        # vertical caps
        for xv in [l, h]:
            ax.plot([xv, xv], [yi-0.15, yi+0.15], color=colour, linewidth=1.5, zorder=3)

        # annotation
        sig_label = f"p={wp:.4f}" + (" *" if wp < 0.05 else "")
        ax.text(h + 0.004, yi, f"{w_}/8 wins  {sig_label}",
                va="center", fontsize=8.5, color=colour)

    ax.axvline(0, color="#888888", linewidth=1.2, linestyle="--", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean Δ Macro F2 (A − B)  with 95% Bootstrap CI")
    ax.set_title("Fig 6 — Pairwise Statistical Robustness Tests (n=8 projects)",
                 fontweight="bold", pad=10)
    ax.set_xlim(-0.04, 0.27)

    sig_dot = plt.Line2D([0],[0], marker="o", color=C["sig"],   linestyle="None",
                          markersize=8, label="CI excludes zero (robust)")
    ns_dot  = plt.Line2D([0],[0], marker="D", color=C["neutral"], linestyle="None",
                          markersize=8, label="CI crosses zero (not robust)")
    ax.legend(handles=[sig_dot, ns_dot], loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out = FIGURES / "Fig6_significance_forest.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf",".png"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — SIMILARITY INVERSION (pos_sim vs neg_sim, test split)
# ─────────────────────────────────────────────────────────────────────────────
def fig7_similarity_inversion():
    # Try local path first, fall back to Thesis Work path
    local_mine = Path("DATA/.GROUND_TRUTH/hard_negative_mining_metadata_v3.json")
    thesis_mine = Path("DATA/.GROUND_TRUTH/hard_negative_mining_metadata_v3.json")
    mine_path = local_mine if local_mine.exists() else thesis_mine
    mine = load(mine_path)

    pos_sims = [mine["results"][p]["test"]["pos_sim_mean"] for p in PROJECTS]
    neg_sims = [mine["results"][p]["test"]["neg_sim_mean"] for p in PROJECTS]

    x = np.arange(len(PROJECTS))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w/2, pos_sims, w, color="#1565C0", label="True positive sim (source ↔ linked target)")
    b2 = ax.bar(x + w/2, neg_sims, w, color="#B71C1C", label="Hard negative sim (source ↔ nearest non-linked)")

    for bar, v in zip(b1, pos_sims):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.001, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    for bar, v in zip(b2, neg_sims):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.001, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#B71C1C")

    # annotate inverted projects
    for i, (p, n) in enumerate(zip(pos_sims, neg_sims)):
        if n > p:
            ax.annotate("inv.", (x[i] + w/2, n + 0.012), ha="center",
                        fontsize=7.5, color="#B71C1C", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(PROJECTS)
    ax.set_ylim(0.70, 0.96)
    ax.set_ylabel("Mean Cosine Similarity (Qwen3-Embedding-4B)")
    ax.set_title("Fig 7 — Similarity Inversion: Hard Negatives vs True Positives (Test Split)",
                 fontweight="bold", pad=10)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    out = FIGURES / "Fig7_similarity_inversion.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf",".png"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — PRECISION–RECALL SCATTER (LLM methods)
# ─────────────────────────────────────────────────────────────────────────────
def fig8_pr_scatter():
    points = [
        # (label,                    P,      R,      marker, colour,      size)
        ("Zero-Shot Gemma",         0.4399, 0.8148, "o",    C["zeroshot"],  80),
        ("GPT-5.4",                 0.5157, 0.7704, "s",    C["cloud"],    80),
        ("Claude Sonnet 4-6",       0.5255, 0.8498, "^",    C["cloud"],    80),
        ("RAG-A MPNet 2+2",         0.5910, 0.8602, "D",    C["rag_light"],80),
        ("RAG-B Qwen3 2+2",         0.6202, 0.8491, "D",    C["rag"],      90),
        ("RAG-C Qwen3 4+0",         0.5007, 0.9053, "D",    C["rag_light"],80),
        ("RAG-D Hybrid 2+2",        0.6063, 0.8241, "D",    C["rag_light"],80),
        ("LoRA V4",                 0.6872, 0.7757, "P",    C["lora"],     90),
        ("Combined V4 + RAG-B",     0.7170, 0.8212, "*",    C["combined"], 130),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, p, r, marker, colour, sz in points:
        ax.scatter(r, p, marker=marker, color=colour, s=sz, zorder=4,
                   edgecolors="white", linewidths=0.5)
        # nudge labels to avoid overlap
        nudge_x, nudge_y = 0.004, 0.004
        if "Combined" in label: nudge_y = 0.008
        if "RAG-C"   in label: nudge_x = -0.068
        ax.annotate(label, (r + nudge_x, p + nudge_y), fontsize=7.8)

    # F2 iso-lines
    p_vals = np.linspace(0.05, 1.0, 300)
    for f2_iso in [0.65, 0.70, 0.75, 0.80]:
        # F2 = 5pr/(4p+r) → r = 4p*F2 / (5p - F2)
        r_vals = []
        valid_p = []
        for pv in p_vals:
            denom = 5*pv - f2_iso
            if denom <= 0: continue
            rv = 4*pv*f2_iso / denom
            if 0 < rv <= 1:
                r_vals.append(rv)
                valid_p.append(pv)
        if r_vals:
            ax.plot(r_vals, valid_p, "--", color="#AAAAAA", linewidth=0.8, zorder=1)
            ax.text(r_vals[-1]+0.005, valid_p[-1], f"F₂={f2_iso}",
                    fontsize=7.5, color="#999999", va="center")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.68, 0.98)
    ax.set_ylim(0.35, 0.82)
    ax.set_title("Fig 8 — Precision–Recall Trade-off with F₂ Iso-lines (LLM Methods)",
                 fontweight="bold", pad=10)

    # custom legend for marker types
    legend_items = [
        plt.Line2D([0],[0], marker="o", color=C["zeroshot"], linestyle="None", ms=7, label="Zero-Shot LLM"),
        plt.Line2D([0],[0], marker="s", color=C["cloud"],    linestyle="None", ms=7, label="Cloud API"),
        plt.Line2D([0],[0], marker="^", color=C["cloud"],    linestyle="None", ms=7, label="Cloud Batch"),
        plt.Line2D([0],[0], marker="D", color=C["rag"],      linestyle="None", ms=7, label="RAG"),
        plt.Line2D([0],[0], marker="P", color=C["lora"],     linestyle="None", ms=8, label="LoRA"),
        plt.Line2D([0],[0], marker="*", color=C["combined"], linestyle="None", ms=10, label="Combined"),
    ]
    ax.legend(handles=legend_items, loc="lower left", framealpha=0.9)

    plt.tight_layout()
    out = FIGURES / "Fig8_pr_scatter.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf",".png"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — RELIABILITY (parse failure rate + clean vs conservative delta)
# ─────────────────────────────────────────────────────────────────────────────
def fig9_reliability():
    rag_data  = load(RESULTS / "RAG_STAGE2_HYBRID_V3_8192" / "MASTER_RAG_ABCD_COMPARISON.json")
    lora_data = load(RESULTS / "LORA_RERUN_V3" / "MASTER_COMPARISON.json")

    methods, fail_pcts, f2_deltas = [], [], []

    # RAG methods
    for cfg, clabel in [("RAG_A","RAG-A"),("RAG_B","RAG-B"),("RAG_C","RAG-C"),("RAG_D","RAG-D")]:
        c   = rag_data["configs"][cfg]
        fp  = c["deployment_rollup"]["macro_format_failure_rate_pct"]
        d   = c["macro_clean"]["f2"] - c["macro_conservative"]["f2"]
        methods.append(clabel); fail_pcts.append(fp); f2_deltas.append(round(d,4))

    # LoRA methods
    for vk, vlabel in [("V1_NAIVE","V1"),("V2_BALANCED","V2"),("V3_STABILIZED","V3"),
                        ("V4_EFFICIENCY","V4★"),("V5_SYNTHESIS","V5"),("V6_MLP","V6")]:
        v   = lora_data["versions"][vk]
        tot = 10668
        fp  = round(v["total_failed_parses"] / tot * 100, 3)
        d   = v["macro_clean"]["f2"] - v["macro_conservative"]["f2"]
        methods.append(f"LoRA {vlabel}"); fail_pcts.append(fp); f2_deltas.append(round(d,4))

    x = np.arange(len(methods))
    colours_r = ([C["rag_light"]]*2 + [C["rag"]]*1 + [C["rag_light"]] +
                 [C["lora_light"]]*3 + [C["lora"]] + [C["lora_light"]]*2)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(x, fail_pcts, 0.55, color=colours_r, label="Parse fail %", zorder=3)
    ax2.plot(x, f2_deltas, "ko--", markersize=6, linewidth=1.5,
             label="Clean − Conservative F2 (reliability gap)", zorder=4)

    for bar, v in zip(bars, fail_pcts):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.2f}%",
                 ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha="right")
    ax1.set_ylabel("Parse Failure Rate (%)")
    ax2.set_ylabel("Clean F2 − Conservative F2")
    ax1.set_ylim(0, 4.5)
    ax2.set_ylim(0, 0.04)

    lines1, ll1 = ax1.get_legend_handles_labels()
    lines2, ll2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, ll1+ll2, loc="upper left", framealpha=0.9)

    ax1.set_title("Fig 9 — Output Reliability: Parse Failure Rate & Clean vs Conservative F2 Gap (RQ3)",
                  fontweight="bold", pad=10)
    plt.tight_layout()
    out = FIGURES / "Fig9_reliability.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf",".png"))
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("THESIS FIGURE GENERATOR")
    print("="*60)
    print(f"Output directory: {FIGURES.resolve()}\n")

    print("Generating Fig 1 — Main Comparison...")
    fig1_main_comparison()

    print("Generating Fig 2 — Per-Project Heatmap...")
    fig2_heatmap()

    print("Generating Fig 3 — RAG Ablation...")
    fig3_rag_ablation()

    print("Generating Fig 4 — LoRA Ablation...")
    fig4_lora_ablation()

    print("Generating Fig 5 — Combined Breakdown...")
    fig5_combined()

    print("Generating Fig 6 — Significance Forest Plot...")
    fig6_significance()

    print("Generating Fig 7 — Similarity Inversion...")
    fig7_similarity_inversion()

    print("Generating Fig 8 — Precision-Recall Scatter...")
    fig8_pr_scatter()

    print("Generating Fig 9 — Reliability Chart...")
    fig9_reliability()

    print("\n" + "="*60)
    print(f"ALL DONE — 9 figures saved to {FIGURES.resolve()}")
    print("Each figure saved as both .pdf (for LaTeX) and .png (for Word/preview)")
    print("="*60)
