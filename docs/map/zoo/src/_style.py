"""Shared light Art Nouveau style for the Map figures (tokens from docs/DESIGN.md)."""
PAPER, PAPER2, PAPER3 = "#f3eadb", "#ebdfc8", "#e0d2b6"
INK, INK2, LINE = "#2a211b", "#5a4a3d", "#b89f78"
TEAL, GOLD, ROSE, PLUM, SAGE, COPPER = "#246b66", "#b8892e", "#b05a4c", "#6b4a7a", "#6f8f4e", "#9c6b3f"
PALETTE = [TEAL, GOLD, ROSE, PLUM, SAGE, COPPER]
STAGE = {"transform": TEAL, "predict": SAGE, "quantize": ROSE, "entropy": PLUM, "perceptual": GOLD, "prior": COPPER}


def cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("cream_teal", [PAPER, "#9cc4b8", TEAL])


def apply():
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "grid.color": LINE, "grid.alpha": 0.4,
        "legend.facecolor": PAPER, "legend.edgecolor": LINE,
        "axes.prop_cycle": cycler(color=PALETTE),
        "image.cmap": "YlGnBu",
    })
