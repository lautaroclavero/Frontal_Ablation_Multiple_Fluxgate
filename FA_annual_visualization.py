# -*- coding: utf-8 -*-
"""
Plot annual frontal ablation results.
Figure saved to output/figures/Figure_annual.png
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

# CONFIG GLOBAL
sns.set_theme(style="ticks")
sns.set_context("talk", font_scale=1, rc={"lines.linewidth": 2.5})
mpl.rcParams['figure.dpi'] = 300

SCRIPT_DIR = Path(__file__).parent

output_dir = SCRIPT_DIR / "output" / "annual"
df_flow = pd.read_csv(output_dir / "annual_flowlines_results.csv")
df_annual = pd.read_csv(output_dir / "annual_frontal_ablation.csv")

# Filter outliers
df_flow_filtered = df_flow[(df_flow['Ut'] >= 0) & 
                          (df_flow['Uc'] >= 0) & 
                          (df_flow['dL/dt'] > -200)].reset_index(drop=True)

df_annual_sorted = df_annual.sort_values(by='Year')

palette_sur = sns.blend_palette(["lightgreen", "darkgreen"], n_colors=5)
palette_centro = sns.blend_palette(["lightcoral", "red", "lightcoral"], n_colors=5)
palette_centro[2] = "darkred"
palette_norte = sns.blend_palette(["lightblue", "blue"], n_colors=5)[::-1]
paleta_completa = palette_sur + palette_centro + palette_norte

tick_size_x = 22
tick_size_y = 22
x_pad = 0.5

y_pad = {'Ut': 10, 'dL/dt': 50, 'Uc': 50, 'Ht': 10, 'FA': 0.5}

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(20, 20), sharex=True, dpi=300)

sns.lineplot(x="Year", y="Ut", hue="ID", palette=paleta_completa,
             data=df_flow_filtered, marker='o', ax=axes[0], legend=False, linewidth=2)
axes[0].set_ylabel("$U_{T}$ (m a$^{-1}$)", fontsize=22)
axes[0].yaxis.set_tick_params(labelsize=tick_size_y)
axes[0].set_ylim(df_flow_filtered['Ut'].min() - y_pad['Ut'], 
                 df_flow_filtered['Ut'].max() + y_pad['Ut'])
axes[0].text(0.02, 0.85, 'Velocity at glacier terminus', transform=axes[0].transAxes,
             fontsize=22, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))

sns.lineplot(x="Year", y="dL/dt", hue="ID", palette=paleta_completa,
             data=df_flow_filtered, marker='o', ax=axes[1], legend=False, linewidth=2)
axes[1].set_ylabel("dL/dt (m a$^{-1}$)", fontsize=22)
axes[1].yaxis.set_tick_params(labelsize=tick_size_y)
axes[1].set_ylim(df_flow_filtered['dL/dt'].min() - y_pad['dL/dt'], 
                 df_flow_filtered['dL/dt'].max() + y_pad['dL/dt'])
axes[1].text(0.02, 0.85, 'Ice front displacement rate', transform=axes[1].transAxes,
             fontsize=22, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))

sns.lineplot(x="Year", y="Uc", hue="ID", palette=paleta_completa,
             data=df_flow_filtered, marker='o', ax=axes[2], legend=False, linewidth=2)
axes[2].set_ylabel("$\\dot{a}$ (m a$^{-1}$)", fontsize=22)
axes[2].yaxis.set_tick_params(labelsize=tick_size_y)
axes[2].set_ylim(df_flow_filtered['Uc'].min() - y_pad['Uc'], 
                 df_flow_filtered['Uc'].max() + y_pad['Uc'])
axes[2].text(0.02, 0.85, 'Frontal ablation rate', transform=axes[2].transAxes,
             fontsize=22, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))

sns.lineplot(x="Year", y="Ht", hue="ID", palette=paleta_completa,
             data=df_flow_filtered, marker='o', ax=axes[3], legend=False, linewidth=2)
axes[3].set_ylabel("H (m)", fontsize=22)
axes[3].yaxis.set_tick_params(labelsize=tick_size_y)
axes[3].set_ylim(df_flow_filtered['Ht'].min() - y_pad['Ht'], 
                 df_flow_filtered['Ht'].max() + y_pad['Ht'])
axes[3].text(0.02, 0.85, 'Thickness at glacier terminus', transform=axes[3].transAxes,
             fontsize=22, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))

x = df_annual_sorted['Year']
y = df_annual_sorted['Frontal_ablation_Mm3']
error = df_annual_sorted['Err_Frontal_ablation_Mm3']

axes[4].errorbar(x, y, yerr=error, fmt='o', linestyle='--', linewidth=2,
                 markersize=8, color='k', capsize=5, capthick=2)
axes[4].fill_between(x, y-error, y+error, color='gray', alpha=0.5)
axes[4].axhline(y=y.mean(), color='red', linestyle='-', linewidth=2)
axes[4].set_ylabel('FA (Mm$^{3}$ a$^{-1}$)', fontsize=22)
axes[4].yaxis.set_tick_params(labelsize=tick_size_y)
axes[4].set_ylim(1, 7)
axes[4].text(0.02, 0.85, 'Frontal ablation', transform=axes[4].transAxes,
             fontsize=22, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))
axes[4].tick_params(axis='x', labelsize=22)
axes[4].set_xlim(x.min() - x_pad, x.max() + x_pad)

for ax in axes[:4]:
    ax.tick_params(axis='x', labelbottom=False)

plt.tight_layout(rect=[0, 0, 0.88, 1])

output_fig_dir = SCRIPT_DIR / "output" / "figures"
output_fig_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_fig_dir / "Figure_annual.png", dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_fig_dir / 'Figure_annual.png'}")