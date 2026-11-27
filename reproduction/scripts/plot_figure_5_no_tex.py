import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

import os

from analytics.plot_figure_3_and_4 import load_csv_files_to_dataframe, filter_name, improve_legend, X_AXIS_TEXT

X_AXIS_TEXT = "Desired reduction in unwanted content (%)"

sns.set_theme(font_scale=1.0,
        style="ticks",
        rc={
        "text.usetex": False,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
    }, palette="crest")

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)
args = parser.parse_args()

df = load_csv_files_to_dataframe(args.file, rescale=False)

df["beta"] = df.beta.apply(lambda x : f"W% > {str(round(x*100))}%" if x != -1 else "None")
df["Strategy"] = df.apply(lambda x: f"{x.conformal_method.capitalize()}" if x.Method == "Conformal" else x.Method, axis=1)
df["Risk Control"] = df.apply(lambda x: f"{x.conformal_score}" if x.Method == "Conformal" else "None", axis=1)

df["conformal_score"] = df["conformal_score"].fillna("None")
df["conformal_score"] = df.conformal_score.apply(lambda x : "Similarity" if x=="weights" else x)
df["conformal_method"] = df["conformal_method"].fillna("None")

df["Method"] = df["Method"].apply(lambda x: "None" if x == "Classic" else x)
df["Method"] = df["Method"].apply(lambda x: "Pre-train" if x == "Classic (masked)" else x)
df["Strategy"] = df.apply(lambda x: f"{x.conformal_method.capitalize()}" if x.Method == "Conformal" else x.Method, axis=1)
df["Risk Control"] = df.apply(lambda x: x.score_method.capitalize() if x.Method == "Conformal" else "None", axis=1)
df["Strategy"] = df["Strategy"].apply(lambda x: f"Remove" if x=="Remove" else "Replace (Ours)")

def compute_diff(group):
    group = group.copy()
    group["difference"] = np.clip(group["alpha"]-group["empr_harmfulness"], 0, 100)
    return group[["difference", "alpha"]]

tmp = df.groupby(["run_id", "beta", "Risk Control"]).apply(
    compute_diff, include_groups=False
).reset_index()

df.rename(
    columns={"alpha": "% of undesired items",
             "|S|": r"$|S(X)|$",
             "random_items": "# of replaced items"},
    inplace=True
)

FIGURE_SIZE = (3.2,2)

df = df[((df.Strategy == "Replace (Ours)"))]
df = df[df["Risk Control"].isin(["Lightgcl"])]

fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)
g = sns.lineplot(
    data=df,
    x="% of undesired items",
    y="nDCG @ k",
    hue="beta",
    hue_order=["None", "W% > 0%", "W% > 50%", "W% > 100%"],
    markers=True,
    errorbar="sd",
    ax = ax,
    palette="crest"
)
ax.grid(axis="y")
ax.legend(fontsize='x-small')
ax.set_ylabel(r"nDCG @ 20", fontsize="small")
ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
plt.savefig("00_beta_ndcg.pdf", format="pdf", bbox_inches='tight')
plt.clf()

fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)
g = sns.lineplot(
    data=df,
    x="% of undesired items",
    y="Recall @ k",
    hue_order=["None", "W% > 0%", "W% > 50%", "W% > 100%"],
    hue="beta",
    markers=True,
    legend=True,
    errorbar="sd",
    ax=ax,
    palette="crest"
)
ax.grid(axis="y")
ax.legend(fontsize='x-small')
ax.set_ylabel(r"Recall @ 20", fontsize="small")
ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
plt.savefig("00_beta_recall.pdf", format="pdf", bbox_inches='tight')
plt.clf()

fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)
g = sns.lineplot(
    data=df,
    x="% of undesired items",
    y="empr_harmfulness",
    hue_order=["None", "W% > 0%", "W% > 50%", "W% > 100%"],
    hue="beta",
    markers=False,
    legend=True,
    errorbar="sd",
    ax=ax,
    palette="crest"
)
ax.grid(axis="y")
ax.legend(fontsize='x-small')
ax.plot([0, 100], [0, 100], linestyle=':', color='black', label="Optim.")
ax.set(ylim=(105, -5.0))
ax.set_ylabel("Empirical reduction (%)", fontsize="small")
ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
plt.savefig("00_beta_harm.pdf", format="pdf", bbox_inches='tight')
plt.clf()

fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)
g = sns.lineplot(
    data=df,
    x="% of undesired items",
    y=r"$|S(X)|$",
    hue_order=["None", "W% > 0%", "W% > 50%", "W% > 100%"],
    #style="beta",
    #hue="Risk Control",
    hue="beta",
    markers=True,
    legend=True,
    errorbar="sd",
    ax=ax,
    palette="crest"
)
ax.set_ylabel(r"$|S_\lambda(U)|$", fontsize="small")
ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
ax.grid(axis="y")
ax.legend(fontsize='x-small')
plt.savefig("00_beta_size.pdf", format="pdf", bbox_inches='tight')
plt.clf()

fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)
g = sns.lineplot(
    data=df,
    x="% of undesired items",
    y="# of replaced items",
    hue_order=["None", "W% > 0%", "W% > 50%", "W% > 100%"],
    #style="beta",
    #hue="Risk Control",
    hue="beta",
    markers=True,
    legend=True,
    errorbar="sd",
    ax=ax,
    palette="crest"
)
ax.set_ylabel("# of previously seen items", fontsize="small")
ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
ax.grid(axis="y")
ax.legend(fontsize='x-small')
plt.savefig("00_beta_replacement.pdf", format="pdf", bbox_inches='tight')
plt.clf()
