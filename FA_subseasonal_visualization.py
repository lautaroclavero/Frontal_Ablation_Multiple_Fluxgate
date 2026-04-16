# -*- coding: utf-8 -*-
"""
Plot timeseries at flowline level and for the entire glacier.
Figure saved to output/figures/Figure_subseasonal.png
"""

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
from pathlib import Path

# CONFIG GLOBAL
sns.set_theme(style="ticks")
sns.set_context("talk", font_scale=1)
plt.rcParams['figure.dpi'] = 300

SCRIPT_DIR = Path(__file__).parent

df_flow = pd.read_csv(SCRIPT_DIR / 'output/subseasonal/flowlines_results.csv',
                      parse_dates=['Start_Date', 'End_Date'])

df_FA = pd.read_csv(SCRIPT_DIR / 'output/subseasonal/frontal_ablation_results.csv',
                    parse_dates=['Start', 'End'])

df_FA['upper'] = df_FA['Frontal_ablation_m3d'] + df_FA['Err_Frontal_ablation']
df_FA['lower'] = df_FA['Frontal_ablation_m3d'] - df_FA['Err_Frontal_ablation']

palette_sur = sns.blend_palette(["lightgreen", "darkgreen"], n_colors=5)
palette_centro = sns.blend_palette(["lightcoral", "red", "lightcoral"], n_colors=5)
palette_centro[2] = "darkred"
palette_norte = sns.blend_palette(["lightblue", "blue"], n_colors=5)[::-1]
paleta_completa = palette_sur + palette_centro + palette_norte

fig, axs = plt.subplots(5, 1, sharex=True, figsize=(12, 15), dpi=300)

start_date = datetime(2017, 1, 30)
end_date = datetime(2024, 5, 1)

variables = ["Ut", "dL/dt", "Uc", "Ht"]
y_labels = ["$U_{T}$ (m d$^{-1}$)", "dL/dt (m d$^{-1}$)",
            "$\\dot{a}$ (m d$^{-1}$)", "H (m)", "FA (m$^{3}$ d$^{-1}$)"]

y_lims = [(0.1, 0.5), (-2, 1), (0, 2), (0, 100), None]

for i, var in enumerate(variables):
    ax = axs[i]
    ax.set_xlim(start_date, end_date)

    for _, row in df_flow.iterrows():
        color = paleta_completa[row['ID'] - 1]
        ax.hlines(y=row[var], xmin=row['Start_Date'], xmax=row['End_Date'],
                  color=color, linewidth=3)

    if y_lims[i]:
        ax.set_ylim(y_lims[i])

    ax.set_ylabel(y_labels[i], fontsize=20)
    ax.tick_params(axis='y', labelsize=17)

    for year in range(2017, 2025):
        ax.axvline(datetime(year, 12, 31), color='k',
                   linestyle='dashed', linewidth=1, alpha=0.5)

    if i < 4:
        ax.xaxis.set_visible(False)

# FA subplot
ax = axs[4]

for _, row in df_FA.iterrows():
    ax.plot([row['Start'], row['End']],
            [row['Frontal_ablation_m3d'], row['Frontal_ablation_m3d']],
            color='black', linewidth=2)

    ax.fill_between([row['Start'], row['End']],
                    [row['lower'], row['lower']],
                    [row['upper'], row['upper']],
                    color='gray', alpha=0.4)

ax.set_xlim(start_date, end_date)
ax.set_ylim(0, 40000)
ax.set_yticks([0, 10000, 20000, 30000, 40000])

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((3, 3))
ax.yaxis.set_major_formatter(formatter)
ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
ax.yaxis.get_offset_text().set_size(16)

ax.set_ylabel(y_labels[4], fontsize=20)
ax.tick_params(axis='y', labelsize=17)

for year in range(2017, 2025):
    ax.axvline(datetime(year, 12, 31), color='k', ls='dashed', lw=1, alpha=0.5)

axs[4].xaxis.set_minor_locator(mdates.MonthLocator([4, 8, 12]))
axs[4].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
axs[4].xaxis.set_major_locator(mdates.YearLocator())
axs[4].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs[4].tick_params(axis='x', which='major', labelsize=18, pad=18)
axs[4].tick_params(axis='x', which='minor', labelsize=14)

plt.subplots_adjust(hspace=0.05)

plt.tight_layout(rect=[0, 0, 0.88, 1])
output_fig_dir = SCRIPT_DIR / "output" / "figures"
output_fig_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_fig_dir / "Figure_subseasonal.png", dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_fig_dir / 'Figure_subseasonal.png'}")
