"""
Thesis Figure Generator V2
==========================

RQ-structured figure generator for the final thesis results.

Design goals:
  - no figure titles/captions inside the image (LaTeX provides captions)
  - no hard-coded baseline results; latest JSON files are loaded
  - output goes to RESULTS/FIGURES_V2
  - each figure is saved as PNG and PDF
  - uses Pillow only, so it runs in the bundled Codex Python environment

Usage:
    python 11_generate_thesis_figures_v2.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "RESULTS"
FIGURES = RESULTS / "FIGURES_V2"
FIGURES.mkdir(parents=True, exist_ok=True)

PROJECTS = ["AAH", "BEAM", "CB", "FH", "JBIDE", "KEYCLOAK", "KOGITO", "PROJQUAY"]

COLORS = {
    "bg": "#FFFFFF",
    "axis": "#263238",
    "grid": "#D9DEE2",
    "text": "#1F2933",
    "muted": "#697386",
    "baseline": "#8E8E8E",
    "zeroshot": "#4F7DD1",
    "cloud": "#7B1FA2",
    "rag": "#26A69A",
    "rag_light": "#80CBC4",
    "lora": "#EF6C00",
    "lora_light": "#FFCC80",
    "combined": "#1B5E20",
    "danger": "#C62828",
    "neutral": "#455A64",
    "other": "#B0BEC5",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest(pattern: str) -> Path:
    matches = sorted(RESULTS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No result file matched {pattern!r} in {RESULTS}")
    return matches[0]


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def interp_color(c1: str, c2: str, t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    a, b = hex_to_rgb(c1), hex_to_rgb(c2)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()


F_SMALL = font(22)
F_MED = font(26)
F_AXIS = font(30)
F_LABEL = font(32)
F_LABEL_BOLD = font(32, bold=True)
F_NUM = font(24)
F_NUM_BOLD = font(24, bold=True)


def new_canvas(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w, h), hex_to_rgb(COLORS["bg"]))


def save_both(img: Image.Image, name: str):
    png = FIGURES / f"{name}.png"
    pdf = FIGURES / f"{name}.pdf"
    img.save(png)
    img.save(pdf, "PDF", resolution=300.0)
    print(f"  saved {png.name} and {pdf.name}")


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_text_center(draw, xy, text, fnt, fill):
    x, y = xy
    w, h = text_size(draw, text, fnt)
    draw.text((x - w / 2, y - h / 2), text, font=fnt, fill=fill)


def draw_rotated_text(base: Image.Image, xy, text: str, fnt, fill, angle=90):
    tmp_w, tmp_h = text_size(ImageDraw.Draw(base), text, fnt)
    tmp = Image.new("RGBA", (tmp_w + 12, tmp_h + 12), (255, 255, 255, 0))
    d = ImageDraw.Draw(tmp)
    d.text((6, 6), text, font=fnt, fill=fill)
    rot = tmp.rotate(angle, expand=True)
    base.paste(rot, (int(xy[0] - rot.width / 2), int(xy[1] - rot.height / 2)), rot)


def fmt(v: float, n: int = 3) -> str:
    return f"{v:.{n}f}"


def f2_value(metric_obj: Dict) -> float:
    return float(metric_obj["f2"])


def macro_from_baseline(data: Dict) -> Dict[str, float]:
    s = data["summary"]
    return {
        "precision": float(s["avg_precision"]),
        "recall": float(s["avg_recall"]),
        "f1": float(s["avg_f1"]),
        "f2": float(s["avg_f2"]),
    }


def load_all_metrics():
    paths = {
        "vsm": latest("vsm_final_pairs_results*.json"),
        "sbert": latest("sbert_final_pairs_results*.json"),
        "frozen": latest("frozen_bert_final_pairs_results*.json"),
        "zero": RESULTS / "ZERO_SHOT_QWEN_HARD" / "zero_shot_qwen_hard_results.json",
        "gpt": RESULTS / "OPENAI_ZERO_SHOT_BATCH_V3" / "gpt-5_4_merged_matched_single_user_prompt_v1_batch" / "results.json",
        "claude": RESULTS / "CLAUDE_ZERO_SHOT_BATCH_V3" / "claude-sonnet-4-6_matched_single_user_prompt_v1_batch" / "results.json",
        "rag": RESULTS / "RAG_STAGE2_HYBRID_V3_8192" / "MASTER_RAG_ABCD_COMPARISON.json",
        "lora": RESULTS / "LORA_RERUN_V3" / "MASTER_COMPARISON.json",
        "lora_v4": RESULTS / "LORA_RERUN_V3" / "V4_EFFICIENCY" / "results.json",
        "combined": RESULTS / "COMBINED_RERUN_V3" / "MASTER_COMBINED_COMPARISON.json",
        "sig": RESULTS / "significance_tests_clean_f2.json",
    }

    data = {k: load_json(v) for k, v in paths.items()}

    metrics = {
        "VSM": macro_from_baseline(data["vsm"]),
        "SBERT": macro_from_baseline(data["sbert"]),
        "Frozen BERT": macro_from_baseline(data["frozen"]),
        "Gemma zero-shot": data["zero"]["macro_metrics"],
        "GPT-5.4 zero-shot": data["gpt"]["macro_clean"],
        "Claude zero-shot": data["claude"]["macro_clean"],
        "RAG-B": data["rag"]["configs"]["RAG_B"]["macro_clean"],
        "LoRA V4": data["lora"]["versions"]["V4_EFFICIENCY"]["macro_clean"],
        "Combined": data["combined"]["configurations"]["V4_EFFICIENCY_RAG_B_8192"]["macro_clean"],
    }
    return paths, data, metrics


PATHS, DATA, METRICS = load_all_metrics()


def draw_horizontal_bars(
    name: str,
    rows: Sequence[Tuple[str, float, str]],
    x_min: float,
    x_max: float,
    x_label: str = "Macro F2",
    value_decimals: int = 4,
    width: int = 1800,
    height: Optional[int] = None,
):
    if height is None:
        height = 250 + len(rows) * 86
    img = new_canvas(width, height)
    draw = ImageDraw.Draw(img)

    left, right = 430, width - 180
    top, bottom = 70, height - 115
    plot_w = right - left
    plot_h = bottom - top

    ticks = [round(x_min + i * (x_max - x_min) / 5, 2) for i in range(6)]
    for t in ticks:
        x = left + (t - x_min) / (x_max - x_min) * plot_w
        draw.line((x, top, x, bottom), fill=COLORS["grid"], width=2)
        draw_text_center(draw, (x, bottom + 38), f"{t:.2f}", F_SMALL, COLORS["muted"])
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=3)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=3)

    n = len(rows)
    step = plot_h / n
    bar_h = min(46, step * 0.58)
    for i, (label, value, color) in enumerate(rows):
        y = top + i * step + step / 2
        x_val = left + (value - x_min) / (x_max - x_min) * plot_w
        draw.text((left - 22 - text_size(draw, label, F_MED)[0], y - 15), label, font=F_MED, fill=COLORS["text"])
        draw.rounded_rectangle((left, y - bar_h / 2, x_val, y + bar_h / 2), radius=4, fill=color)
        draw.text((x_val + 12, y - 14), fmt(value, value_decimals), font=F_NUM, fill=COLORS["text"])

    draw_text_center(draw, ((left + right) / 2, height - 35), x_label, F_AXIS, COLORS["text"])
    save_both(img, name)


def draw_grouped_bars(
    name: str,
    groups: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], str]],
    y_min: float,
    y_max: float,
    y_label: str = "Score",
    width: int = 1900,
    height: int = 1050,
):
    img = new_canvas(width, height)
    draw = ImageDraw.Draw(img)
    left, right = 170, width - 80
    top, bottom = 80, height - 190
    plot_w = right - left
    plot_h = bottom - top

    for k in range(6):
        v = y_min + (y_max - y_min) * k / 5
        y = bottom - (v - y_min) / (y_max - y_min) * plot_h
        draw.line((left, y, right, y), fill=COLORS["grid"], width=2)
        draw.text((55, y - 15), f"{v:.2f}", font=F_SMALL, fill=COLORS["muted"])
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=3)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=3)
    draw_rotated_text(img, (35, (top + bottom) / 2), y_label, F_AXIS, COLORS["text"], angle=90)

    group_w = plot_w / len(groups)
    bar_w = min(58, group_w / (len(series) + 1.5))
    for gi, group in enumerate(groups):
        cx = left + group_w * (gi + 0.5)
        draw_text_center(draw, (cx, bottom + 45), group, F_MED, COLORS["text"])
        for si, (_, values, color) in enumerate(series):
            offset = (si - (len(series) - 1) / 2) * (bar_w + 10)
            val = values[gi]
            x0, x1 = cx + offset - bar_w / 2, cx + offset + bar_w / 2
            y = bottom - (val - y_min) / (y_max - y_min) * plot_h
            draw.rounded_rectangle((x0, y, x1, bottom), radius=4, fill=color)
            draw_rotated_text(img, ((x0 + x1) / 2, y - 36), fmt(val, 3), F_SMALL, COLORS["text"], angle=90)

    # legend
    lx, ly = left, 25
    for label, _, color in series:
        draw.rounded_rectangle((lx, ly, lx + 34, ly + 22), radius=3, fill=color)
        draw.text((lx + 46, ly - 3), label, font=F_SMALL, fill=COLORS["text"])
        lx += 280

    save_both(img, name)


def fig_rq1_baselines_vs_zero():
    rows = [
        ("Gemma zero-shot", METRICS["Gemma zero-shot"]["f2"], COLORS["zeroshot"]),
        ("Frozen BERT", METRICS["Frozen BERT"]["f2"], COLORS["baseline"]),
        ("SBERT", METRICS["SBERT"]["f2"], COLORS["baseline"]),
        ("VSM", METRICS["VSM"]["f2"], COLORS["baseline"]),
    ]
    rows.sort(key=lambda x: x[1], reverse=True)
    draw_horizontal_bars("RQ1_baselines_vs_gemma_zero_shot", rows, 0.48, 0.74)


def fig_rq1_cloud():
    rows = [
        ("Claude Sonnet 4.6", METRICS["Claude zero-shot"]["f2"], COLORS["cloud"]),
        ("GPT-5.4", METRICS["GPT-5.4 zero-shot"]["f2"], COLORS["cloud"]),
        ("Gemma 4 31B", METRICS["Gemma zero-shot"]["f2"], COLORS["zeroshot"]),
    ]
    rows.sort(key=lambda x: x[1], reverse=True)
    draw_horizontal_bars("RQ1_1_zero_shot_cloud_comparison", rows, 0.65, 0.78, height=560)


def fig_rq2_champions():
    rows = [
        ("Combined", METRICS["Combined"]["f2"], COLORS["combined"]),
        ("RAG-B", METRICS["RAG-B"]["f2"], COLORS["rag"]),
        ("LoRA V4", METRICS["LoRA V4"]["f2"], COLORS["lora"]),
        ("Gemma zero-shot", METRICS["Gemma zero-shot"]["f2"], COLORS["zeroshot"]),
    ]
    draw_horizontal_bars("RQ2_champion_methods", rows, 0.66, 0.82, height=660)


def fig_rag_ablation():
    rag = DATA["rag"]["configs"]
    rows = []
    for key, color in [("RAG_B", COLORS["rag"]), ("RAG_A", COLORS["rag_light"]), ("RAG_C", COLORS["rag_light"]), ("RAG_D", COLORS["rag_light"])]:
        rows.append((rag[key]["label"].replace("RAG-", ""), rag[key]["macro_clean"]["f2"], color))
    rows.sort(key=lambda x: x[1], reverse=True)
    draw_horizontal_bars("RQ2_2_rag_ablation_f2", rows, 0.74, 0.80, height=650)


def fig_lora_ablation():
    lora = DATA["lora"]["versions"]
    order = ["V1_NAIVE", "V2_BALANCED", "V3_STABILIZED", "V4_EFFICIENCY", "V5_SYNTHESIS", "V6_MLP"]
    rows = []
    short = {
        "V1_NAIVE": "V1 naive",
        "V2_BALANCED": "V2 balanced",
        "V3_STABILIZED": "V3 stabilized",
        "V4_EFFICIENCY": "V4 efficiency",
        "V5_SYNTHESIS": "V5 synthesis",
        "V6_MLP": "V6 MLP",
    }
    for k in order:
        color = COLORS["lora"] if k == "V4_EFFICIENCY" else COLORS["lora_light"]
        rows.append((short[k], lora[k]["macro_clean"]["f2"], color))
    draw_horizontal_bars("RQ2_2_lora_ablation_f2", rows, 0.60, 0.77, height=820)


def fig_combined_components():
    comb = DATA["combined"]["configurations"]
    groups = ["Precision", "Recall", "F1", "F2", "Accuracy"]
    keys = ["precision", "recall", "f1", "f2", "accuracy"]
    lora = [comb["LORA_V4_ALONE"]["macro_clean"][k] for k in keys]
    rag = [comb["RAG_B_ALONE"]["macro_clean"][k] for k in keys]
    combined = [comb["V4_EFFICIENCY_RAG_B_8192"]["macro_clean"][k] for k in keys]
    draw_grouped_bars(
        "RQ2_3_combined_components",
        groups,
        [
            ("LoRA V4", lora, COLORS["lora"]),
            ("RAG-B", rag, COLORS["rag"]),
            ("Combined", combined, COLORS["combined"]),
        ],
        0.58,
        0.90,
        y_label="Macro score",
    )


def fig_significance():
    sig = DATA["sig"]["comparisons"]
    labels = [c["comparison"].replace("ZeroShot", "Zero-shot") for c in sig]
    means = [c["mean_diff"] for c in sig]
    lows = [c["boot_ci_low"] for c in sig]
    highs = [c["boot_ci_high"] for c in sig]
    wins = [c["wins_A"] for c in sig]
    pvals = [c["wilcoxon_p"] for c in sig]

    w, h = 1900, 850
    img = new_canvas(w, h)
    draw = ImageDraw.Draw(img)
    left, right = 520, w - 260
    top, bottom = 80, h - 110
    x_min, x_max = -0.03, 0.14
    plot_w = right - left

    for t in [-0.02, 0.00, 0.04, 0.08, 0.12]:
        x = left + (t - x_min) / (x_max - x_min) * plot_w
        draw.line((x, top, x, bottom), fill=COLORS["grid"], width=2)
        draw_text_center(draw, (x, bottom + 38), f"{t:+.2f}", F_SMALL, COLORS["muted"])
    x0 = left + (0 - x_min) / (x_max - x_min) * plot_w
    draw.line((x0, top, x0, bottom), fill=COLORS["axis"], width=3)
    draw_text_center(draw, ((left + right) / 2, h - 35), "Mean delta Macro F2 (A - B), 95% bootstrap CI", F_AXIS, COLORS["text"])

    step = (bottom - top) / len(labels)
    for i, label in enumerate(labels):
        y = top + i * step + step / 2
        draw.text((left - 30 - text_size(draw, label, F_MED)[0], y - 14), label, font=F_MED, fill=COLORS["text"])
        l = left + (lows[i] - x_min) / (x_max - x_min) * plot_w
        m = left + (means[i] - x_min) / (x_max - x_min) * plot_w
        r = left + (highs[i] - x_min) / (x_max - x_min) * plot_w
        robust = not (lows[i] <= 0 <= highs[i])
        color = COLORS["danger"] if robust else COLORS["neutral"]
        draw.line((l, y, r, y), fill=color, width=7)
        draw.line((l, y - 18, l, y + 18), fill=color, width=5)
        draw.line((r, y - 18, r, y + 18), fill=color, width=5)
        draw.ellipse((m - 13, y - 13, m + 13, y + 13), fill=color)
        suffix = "*" if pvals[i] < 0.05 else ""
        draw.text((r + 18, y - 14), f"{wins[i]}/8 wins, p={pvals[i]:.4f}{suffix}", font=F_SMALL, fill=color)
    save_both(img, "RQ2_statistical_robustness")


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def fig_latency():
    zero = DATA["zero"]
    rag_b = DATA["rag"]["configs"]["RAG_B"]["deployment_rollup"]
    lora_v4 = DATA["lora_v4"]
    comb = DATA["combined"]["configurations"]["V4_EFFICIENCY_RAG_B_8192"]["deployment_rollup"]
    lora_ms = mean(p["deployment"]["time_per_pair_ms"] for p in lora_v4["per_project"])
    lora_gen = lora_ms
    rows = [
        ("Zero-shot", zero["avg_time_per_pair"] * 1000.0, 0.0, zero["avg_time_per_pair"] * 1000.0, COLORS["zeroshot"]),
        ("LoRA V4", lora_ms, 0.0, lora_gen, COLORS["lora"]),
        ("RAG-B", rag_b["avg_time_per_pair_ms"], rag_b["avg_retrieval_total_ms"], rag_b["avg_generation_ms"], COLORS["rag"]),
        ("Combined", comb["avg_time_per_pair_ms"], comb["avg_retrieval_total_ms"], comb["avg_generation_ms"], COLORS["combined"]),
    ]
    w, h = 1900, 700
    img = new_canvas(w, h)
    draw = ImageDraw.Draw(img)
    left, right = 350, w - 210
    top, bottom = 75, h - 110
    x_max = max(r[1] for r in rows) * 1.12
    plot_w = right - left
    for t in range(0, int(math.ceil(x_max / 500) * 500) + 1, 500):
        x = left + t / x_max * plot_w
        draw.line((x, top, x, bottom), fill=COLORS["grid"], width=2)
        draw_text_center(draw, (x, bottom + 35), f"{t}", F_SMALL, COLORS["muted"])
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=3)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=3)
    step = (bottom - top) / len(rows)
    for i, (label, total, retr, gen, color) in enumerate(rows):
        y = top + i * step + step / 2
        bar_h = 44
        draw.text((left - 22 - text_size(draw, label, F_MED)[0], y - 15), label, font=F_MED, fill=COLORS["text"])
        x_retr = left + retr / x_max * plot_w
        x_gen = left + (retr + gen) / x_max * plot_w
        x_total = left + total / x_max * plot_w
        if retr > 0:
            draw.rounded_rectangle((left, y - bar_h / 2, x_retr, y + bar_h / 2), radius=4, fill=COLORS["rag_light"])
        draw.rounded_rectangle((x_retr, y - bar_h / 2, x_gen, y + bar_h / 2), radius=4, fill=color)
        if x_total > x_gen:
            draw.rounded_rectangle((x_gen, y - bar_h / 2, x_total, y + bar_h / 2), radius=4, fill=COLORS["other"])
        draw.text((x_total + 12, y - 14), f"{total:.0f} ms", font=F_NUM, fill=COLORS["text"])
    draw_text_center(draw, ((left + right) / 2, h - 35), "Single-stream inference time per pair (ms)", F_AXIS, COLORS["text"])
    # legend
    lx, ly = left, 20
    for label, color in [("retrieval", COLORS["rag_light"]), ("generation", COLORS["rag"]), ("overhead", COLORS["other"])]:
        draw.rounded_rectangle((lx, ly, lx + 32, ly + 20), radius=3, fill=color)
        draw.text((lx + 42, ly - 4), label, font=F_SMALL, fill=COLORS["text"])
        lx += 190
    save_both(img, "RQ3_1_latency_breakdown")


def fig_reliability():
    rag = DATA["rag"]["configs"]
    lora = DATA["lora"]["versions"]
    total = 10668
    rows = []
    for k in ["RAG_A", "RAG_B", "RAG_C", "RAG_D"]:
        rows.append((rag[k]["label"].split()[0], rag[k]["deployment_rollup"]["macro_format_failure_rate_pct"], COLORS["rag"]))
    for k in ["V1_NAIVE", "V2_BALANCED", "V3_STABILIZED", "V4_EFFICIENCY", "V5_SYNTHESIS", "V6_MLP"]:
        label = k.split("_")[0]
        rows.append((f"LoRA {label}", lora[k]["total_failed_parses"] / total * 100.0, COLORS["lora"] if k == "V4_EFFICIENCY" else COLORS["lora_light"]))
    draw_horizontal_bars(
        "RQ3_3_parse_failure_rate",
        rows,
        0.0,
        3.2,
        x_label="Parse failure rate (%)",
        value_decimals=2,
        width=1800,
        height=1050,
    )


def fig_heatmap():
    sig = DATA["sig"]["per_project_f2"]
    claude = {r["project"]: r["metrics_clean"]["f2"] for r in DATA["claude"]["per_project"]}
    gpt = {r["project"]: r["metrics_clean"]["f2"] for r in DATA["gpt"]["per_project"]}
    methods = [
        ("Gemma", sig["ZeroShot"]),
        ("GPT-5.4", [gpt[p] for p in PROJECTS]),
        ("Claude", [claude[p] for p in PROJECTS]),
        ("LoRA V4", sig["LoRA-V4"]),
        ("RAG-B", sig["RAG-B"]),
        ("Combined", sig["Combined"]),
    ]
    cell_w, cell_h = 150, 88
    left, top = 260, 90
    w = left + len(PROJECTS) * cell_w + 100
    h = top + len(methods) * cell_h + 125
    img = new_canvas(w, h)
    draw = ImageDraw.Draw(img)
    for j, p in enumerate(PROJECTS):
        draw_text_center(draw, (left + j * cell_w + cell_w / 2, top - 34), p, F_SMALL, COLORS["text"])
    for i, (m, vals) in enumerate(methods):
        y = top + i * cell_h
        draw.text((left - 25 - text_size(draw, m, F_MED)[0], y + cell_h / 2 - 16), m, font=F_MED, fill=COLORS["text"])
        for j, v in enumerate(vals):
            x = left + j * cell_w
            c = interp_color("#F4F8C8", "#006837", (v - 0.55) / (0.95 - 0.55))
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=c)
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            fill = "#FFFFFF" if lum < 120 else COLORS["text"]
            draw_text_center(draw, (x + cell_w / 2, y + cell_h / 2), fmt(v, 3), F_NUM_BOLD, fill)
    draw_text_center(draw, ((left + len(PROJECTS) * cell_w) / 2, h - 40), "Per-project clean F2", F_AXIS, COLORS["text"])
    save_both(img, "SUP_per_project_f2_heatmap")


def write_manifest():
    manifest = {
        "output_dir": str(FIGURES),
        "source_files": {k: str(v) for k, v in PATHS.items()},
        "macro_metrics": {k: {mk: round(float(mv), 4) for mk, mv in v.items() if isinstance(mv, (int, float))} for k, v in METRICS.items()},
        "figures": [
            "RQ1_baselines_vs_gemma_zero_shot",
            "RQ1_1_zero_shot_cloud_comparison",
            "RQ2_champion_methods",
            "RQ2_2_rag_ablation_f2",
            "RQ2_2_lora_ablation_f2",
            "RQ2_3_combined_components",
            "RQ2_statistical_robustness",
            "RQ3_1_latency_breakdown",
            "RQ3_3_parse_failure_rate",
            "SUP_per_project_f2_heatmap",
        ],
        "note": "Figures contain no embedded captions/titles; use LaTeX captions in the thesis.",
    }
    with open(FIGURES / "figure_manifest_v2.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main():
    print(f"Generating thesis figures V2 into {FIGURES}")
    fig_rq1_baselines_vs_zero()
    fig_rq1_cloud()
    fig_rq2_champions()
    fig_rag_ablation()
    fig_lora_ablation()
    fig_combined_components()
    fig_significance()
    fig_latency()
    fig_reliability()
    fig_heatmap()
    write_manifest()
    print("Done.")


if __name__ == "__main__":
    main()
