#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Publication-Ready Figure Generator
Generates camera-ready, high-resolution (300 DPI) figures for the research paper:
  - Figure 1: Architectural Workflow Diagram (2-step MDP Adaptive Routing)
  - Figure 2: Disaggregated Style-by-Style Retrieval Hit@1 & Accuracy Comparison
  - Figure 3: Ablation Analysis (Component Contribution on Hit@1 & Key-Fact Accuracy)
  - Figure 4: Roman Urdu Orthographic Noise Robustness Curve (Stress Test Degradation)

Outputs saved to experiments/figures/ in both PNG (300 DPI) and vector PDF formats.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Set global publication styling (clean, modern, legible)
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Professional Academic Color Palette (Colorblind-friendly Okabe-Ito / Paul Tol)
COLORS = {
    "vanilla": "#56B4E9",      # Light Blue
    "dr_rag": "#E69F00",       # Amber / Orange
    "react": "#999999",        # Neutral Slate
    "dynrag": "#009E73",       # Emerald Green (Ours / Highlight)
    "accent_red": "#D55E00",   # Vermillion / Highlight Red
    "dark_blue": "#0072B2",    # Deep Blue
    "purple": "#CC79A7",       # Soft Purple
}


def create_figure1_architecture(out_dir: Path) -> None:
    """Figure 1: Architectural Flowchart of the 2-step MDP Urdu-DynRAG System."""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.axis("off")

    def draw_box(x, y, w, h, text, bg="#f8f9fa", border="#333333", lw=1.5, title="", title_color="#222222"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor=bg, edgecolor=border, linewidth=lw, zorder=2
        )
        ax.add_patch(rect)
        if title:
            ax.text(x + w / 2, y + h - 0.05, title, ha="center", va="top",
                    fontsize=10.5, fontweight="bold", color=title_color, zorder=3)
            ax.text(x + w / 2, y + (h - 0.05) / 2, text, ha="center", va="center",
                    fontsize=9.5, color="#333333", zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#222222", zorder=3)

    def draw_arrow(x1, y1, x2, y2, label="", color="#444444", lw=1.8, style="-|>"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=3, shrinkB=3),
            zorder=4
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.03, label, ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color=color, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))

    # 1. Input Query
    draw_box(0.02, 0.45, 0.15, 0.22, "Urdu, Roman Urdu,\nor Code-Switched\nQuestion ($q_0$)",
             bg="#EBF5FB", border="#2980B9", lw=2, title="1. User Input", title_color="#1B4F72")

    # 2. State Extractor
    draw_box(0.22, 0.45, 0.17, 0.22, "14-Dim Vector ($s_0$)\n• Script Ratio ($f_2$)\n• Top-1 Retrieval ($f_4,f_5$)\n• Margins ($f_6,f_7$)\n• Entropy ($f_{10}-f_{12}$)",
             bg="#E8F8F5", border="#16A085", lw=2, title="2. State Extractor", title_color="#0E6251")
    draw_arrow(0.17, 0.56, 0.22, 0.56)

    # 3. Policy Network (Double DQN)
    draw_box(0.44, 0.45, 0.17, 0.22, "Double DQN Policy\n$Q(s_0, a)$\nSelects Action $a \\in \\{0,1,2,3\\}$",
             bg="#FEF9E7", border="#D4AC0D", lw=2, title="3. RL Policy ($t=0$)", title_color="#7D6608")
    draw_arrow(0.39, 0.56, 0.44, 0.56)

    # Action 0: Direct Parametric
    draw_box(0.68, 0.76, 0.14, 0.16, "No Retrieval\n($\\mathrm{Cost}=0$)",
             bg="#FADBD8", border="#C0392B", lw=1.8, title="Action 0: Direct", title_color="#78281F")
    draw_arrow(0.53, 0.67, 0.68, 0.84, label="Parametric", color="#C0392B")

    # Action 1: Raw Chunk Retrieval
    draw_box(0.68, 0.54, 0.14, 0.16, "Search Chunks\n($D_{\\mathrm{raw}}$ FAISS)",
             bg="#E8F8F5", border="#27AE60", lw=1.8, title="Action 1: Raw", title_color="#1E8449")
    draw_arrow(0.61, 0.59, 0.68, 0.62, label="Clean Urdu", color="#27AE60")

    # Action 2: QA Index Retrieval
    draw_box(0.68, 0.32, 0.14, 0.16, "Search QA Pairs\n($D_{\\mathrm{qa}}$ FAISS)",
             bg="#EBF5FB", border="#2980B9", lw=1.8, title="Action 2: DR QA", title_color="#1B4F72")
    draw_arrow(0.61, 0.51, 0.68, 0.40, label="High Margin", color="#2980B9")

    # Action 3: Reformulator (Two-step MDP)
    draw_box(0.44, 0.08, 0.17, 0.22, "Transliteration &\nLexicon Normalization\n$q_0 \\rightarrow q_{\\mathrm{ref}}$ (Nastaliq)",
             bg="#F5EEF8", border="#8E44AD", lw=2, title="Action 3: Reformulate", title_color="#512E5F")
    draw_arrow(0.525, 0.45, 0.525, 0.30, label="Roman / Noisy", color="#8E44AD")

    # Transition to Step t=1
    draw_box(0.68, 0.08, 0.14, 0.22, "State $s_1(q_{\\mathrm{ref}}, t=1)$\nPolicy picks:\nTerminal $a \\in \\{1, 2\\}$",
             bg="#FCF3CF", border="#B7950B", lw=2, title="Step $t=1$", title_color="#7D6608")
    draw_arrow(0.61, 0.19, 0.68, 0.19, label="Normalized", color="#8E44AD")

    # Arrow from Step t=1 to terminal retrievers
    draw_arrow(0.82, 0.22, 0.82, 0.54, label="Retrieve on $q_{\\mathrm{ref}}$", color="#8E44AD")

    # 4. Generator Output
    draw_box(0.87, 0.45, 0.11, 0.22, "Qwen2.5-7B/3B\nGreedy ($T=0$)\nAccurate Answer ($y$)",
             bg="#EAFAF1", border="#2ECC71", lw=2, title="4. Generator", title_color="#196F3D")

    draw_arrow(0.82, 0.84, 0.87, 0.63, color="#444444")
    draw_arrow(0.82, 0.62, 0.87, 0.56, color="#444444")
    draw_arrow(0.82, 0.40, 0.87, 0.49, color="#444444")

    ax.set_title("Figure 1: Urdu-DynRAG Two-Step MDP Adaptive Retrieval Architecture", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    fig.savefig(out_dir / "fig1_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig1_architecture.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Figures] Saved Figure 1 to {out_dir / 'fig1_architecture.png'}")


def create_figure2_style_performance(out_dir: Path) -> None:
    """Figure 2: Dual-panel grouped bar chart of Retrieval Hit@1 and Key-Fact Accuracy across styles."""
    styles = ["S1 (Clean Urdu)", "S2 (Conversational)", "S3 (Code-Switched)", "S4 (Roman Urdu)"]
    
    # Hit@1 Data
    hit1_vanilla = [98.8, 92.9, 97.6, 75.3]
    hit1_dr_rag = [92.9, 89.4, 96.5, 74.1]
    hit1_react = [98.8, 92.9, 91.8, 97.6]
    hit1_dynrag = [100.0, 90.6, 90.6, 92.9]

    # Accuracy Data (including Parametric)
    styles_acc = ["S1", "S2", "S3", "S4 (Roman)", "Parametric"]
    acc_vanilla = [76.2, 57.2, 59.6, 54.7, 20.6]
    acc_dr_rag = [72.3, 54.3, 58.8, 53.5, 20.6]
    acc_react = [76.2, 57.2, 57.9, 73.2, 24.4]
    acc_dynrag = [77.4, 54.0, 54.1, 64.3, 31.1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # --- Panel A: Grounded Retrieval Hit@1 (%) ---
    x = np.arange(len(styles))
    width = 0.20

    rects1 = ax1.bar(x - 1.5 * width, hit1_vanilla, width, label="Vanilla Chunk RAG", color=COLORS["vanilla"], edgecolor="#333", lw=0.8)
    rects2 = ax1.bar(x - 0.5 * width, hit1_dr_rag, width, label="DR-RAG (Val-Tuned)", color=COLORS["dr_rag"], edgecolor="#333", lw=0.8)
    rects3 = ax1.bar(x + 0.5 * width, hit1_react, width, label="Zero-Shot ReAct", color=COLORS["react"], edgecolor="#333", lw=0.8)
    rects4 = ax1.bar(x + 1.5 * width, hit1_dynrag, width, label="Urdu-DynRAG (Ours)", color=COLORS["dynrag"], edgecolor="#333", lw=1.2)

    # Highlight Roman Urdu gain
    ax1.annotate(
        "+17.6%",
        xy=(3 + 1.5 * width, 92.9), xytext=(3 + 0.5 * width, 102),
        ha="center", fontsize=10, fontweight="bold", color=COLORS["accent_red"],
        arrowprops=dict(arrowstyle="->", color=COLORS["accent_red"], lw=1.5)
    )

    ax1.set_ylabel("Retrieval Hit@1 (%)")
    ax1.set_title("(a) Retrieval Hit@1 across Query Styles", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(styles, rotation=12, ha="right")
    ax1.set_ylim(50, 108)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)

    # --- Panel B: Key-Fact Accuracy (%) ---
    x_acc = np.arange(len(styles_acc))
    ax2.bar(x_acc - 1.5 * width, acc_vanilla, width, label="Vanilla Chunk RAG", color=COLORS["vanilla"], edgecolor="#333", lw=0.8)
    ax2.bar(x_acc - 0.5 * width, acc_dr_rag, width, label="DR-RAG (Val-Tuned)", color=COLORS["dr_rag"], edgecolor="#333", lw=0.8)
    ax2.bar(x_acc + 0.5 * width, acc_react, width, label="Zero-Shot ReAct", color=COLORS["react"], edgecolor="#333", lw=0.8)
    ax2.bar(x_acc + 1.5 * width, acc_dynrag, width, label="Urdu-DynRAG (Ours)", color=COLORS["dynrag"], edgecolor="#333", lw=1.2)

    # Highlight S4 & Parametric accuracy gains
    ax2.annotate(
        "+9.6%",
        xy=(3 + 1.5 * width, 64.3), xytext=(3 + 1.2 * width, 72),
        ha="center", fontsize=9.5, fontweight="bold", color=COLORS["accent_red"],
        arrowprops=dict(arrowstyle="->", color=COLORS["accent_red"], lw=1.2)
    )
    ax2.annotate(
        "+10.5%",
        xy=(4 + 1.5 * width, 31.1), xytext=(4 + 1.2 * width, 40),
        ha="center", fontsize=9.5, fontweight="bold", color=COLORS["accent_red"],
        arrowprops=dict(arrowstyle="->", color=COLORS["accent_red"], lw=1.2)
    )

    ax2.set_ylabel("Key-Fact Accuracy (%)")
    ax2.set_title("(b) Key-Fact Accuracy across Styles & Parametric", fontweight="bold")
    ax2.set_xticks(x_acc)
    ax2.set_xticklabels(styles_acc)
    ax2.set_ylim(0, 88)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_dir / "fig2_style_performance.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig2_style_performance.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Figures] Saved Figure 2 to {out_dir / 'fig2_style_performance.png'}")


def create_figure3_ablation_impact(out_dir: Path) -> None:
    """Figure 3: Ablation degradation chart showing impact of removing Action 3, D_qa, and Entropy."""
    configs = [
        "Full Urdu-DynRAG",
        "w/o Action 3 (Reformulation)",
        "w/o Action 2 (DR QA Index)",
        "w/o Entropy Features (f10-f12)",
    ]
    hit1_vals = [89.0, 84.5, 89.4, 90.4]
    acc_vals = [59.9, 58.0, 60.7, 60.4]
    f1_vals = [0.474, 0.447, 0.477, 0.473]

    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=300)

    y = np.arange(len(configs))
    height = 0.35

    bar1 = ax1.barh(y - height / 2, hit1_vals, height, label="Grounded Hit@1 (%)", color=COLORS["dynrag"], edgecolor="#333", lw=0.9)
    bar2 = ax1.barh(y + height / 2, acc_vals, height, label="Key-Fact Accuracy (%)", color=COLORS["vanilla"], edgecolor="#333", lw=0.9)

    # Annotate values on bars
    for b in bar1:
        w = b.get_width()
        ax1.text(w - 3.5, b.get_y() + b.get_height() / 2, f"{w:.1f}%", va="center", ha="right", color="white", fontweight="bold", fontsize=9.5)
    for b in bar2:
        w = b.get_width()
        ax1.text(w - 3.5, b.get_y() + b.get_height() / 2, f"{w:.1f}%", va="center", ha="right", color="white", fontweight="bold", fontsize=9.5)

    # Highlight Action 3 collapse
    ax1.annotate(
        "Critical Drop:\n-4.5% Hit@1, -0.027 F1",
        xy=(84.5, 1 - height / 2), xytext=(73, 1 - 0.1),
        ha="center", fontsize=9.5, fontweight="bold", color=COLORS["accent_red"],
        arrowprops=dict(arrowstyle="->", color=COLORS["accent_red"], lw=1.5)
    )

    ax1.set_yticks(y)
    ax1.set_yticklabels(configs, fontweight="medium")
    ax1.invert_yaxis()  # Top-down order
    ax1.set_xlabel("Metric Score (%)")
    ax1.set_xlim(50, 96)
    ax1.set_title("Figure 3: Urdu-DynRAG Component Ablation Study (5-Fold CV Pooled)", fontweight="bold", pad=12)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_ablation_impact.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig3_ablation_impact.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Figures] Saved Figure 3 to {out_dir / 'fig3_ablation_impact.png'}")


def create_figure4_orthographic_robustness(out_dir: Path) -> None:
    """Figure 4: Stress test degradation curve across 5 Roman Urdu noise regimes."""
    regimes = [
        "Clean\nRoman Urdu",
        "Regime A:\nVowel Shifts",
        "Regime B:\nConsonants",
        "Regime C:\nSMS Dropping",
        "Regime D:\nCompound Noise",
    ]
    vanilla_hit1 = [75.3, 80.0, 74.1, 75.3, 65.9]
    dynrag_hit1 = [97.6, 84.7, 83.5, 81.2, 85.9]

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=300)

    x = np.arange(len(regimes))

    # Plot line series with markers
    ax.plot(x, vanilla_hit1, marker="s", markersize=8, linewidth=2.4, color=COLORS["accent_red"], label="Vanilla Chunk RAG (No Reformulation)")
    ax.plot(x, dynrag_hit1, marker="o", markersize=8, linewidth=2.8, color=COLORS["dynrag"], label="Urdu-DynRAG (With Reformulator)")

    # Fill between lines to highlight the robustness gap
    ax.fill_between(x, vanilla_hit1, dynrag_hit1, color=COLORS["dynrag"], alpha=0.12, label="DynRAG Advantage Zone")

    # Annotate specific key deltas
    ax.annotate(
        "+22.4%",
        xy=(0, 97.6), xytext=(0, 101),
        ha="center", fontsize=10, fontweight="bold", color=COLORS["dynrag"]
    )
    ax.annotate(
        "+20.0%\n(Noise Resilient)",
        xy=(4, 85.9), xytext=(4, 91),
        ha="center", fontsize=9.5, fontweight="bold", color=COLORS["dynrag"],
        arrowprops=dict(arrowstyle="->", color=COLORS["dynrag"], lw=1.3)
    )
    ax.annotate(
        "Vanilla Collapse:\n65.9% (-9.4%)",
        xy=(4, 65.9), xytext=(3.6, 59),
        ha="center", fontsize=9.5, fontweight="bold", color=COLORS["accent_red"],
        arrowprops=dict(arrowstyle="->", color=COLORS["accent_red"], lw=1.3)
    )

    ax.set_ylabel("Top-1 Retrieval Hit@1 (%)")
    ax.set_title("Figure 4: Roman Urdu Orthographic Noise Robustness Curve (N=85 queries/regime)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylim(55, 105)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_orthographic_robustness.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig4_orthographic_robustness.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Figures] Saved Figure 4 to {out_dir / 'fig4_orthographic_robustness.png'}")


def main() -> None:
    print("=" * 80)
    print("URDU-DYNRAG: GENERATING PUBLICATION-READY FIGURES")
    print("=" * 80)

    out_dir = Path("experiments/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    create_figure1_architecture(out_dir)
    create_figure2_style_performance(out_dir)
    create_figure3_ablation_impact(out_dir)
    create_figure4_orthographic_robustness(out_dir)

    print("\n[Figures Complete] All 4 publication figures generated successfully in experiments/figures/!")


if __name__ == "__main__":
    main()

