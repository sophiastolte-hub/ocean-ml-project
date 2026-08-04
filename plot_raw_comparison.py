"""
plot_raw_comparison.py

LLC data is filtered to March-July
to match the mooring observation period (April-June 2023).
This makes the seasonal comparison fair.

Output -> /scratch/gpfs/LRGROUP/ds5324/
  raw_comparison_panels.png
  raw_comparison_overlay.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- Paths ---------------------------------------------------------------------
llc_dir  = '/scratch/gpfs/LRGROUP/ds5324/llc_at_moorings/'
moor_dir = '/scratch/gpfs/LRGROUP/bc5848/california_current/data/processed/moorings_interpolated/'
out_dir  = '/scratch/gpfs/LRGROUP/ds5324/'

stations = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7',
            'S1', 'S2', 'S3', 'S4']

# -- Load LLC depth and time arrays --------------------------------------------
llc_depths = np.load(f'{llc_dir}llc_depths.npy')
llc_time   = np.load(f'{llc_dir}llc_time.npy')

# Time is stored as seconds since 2011-09-10, hourly steps.
# Convert to datetime so we can extract the month of each time step.
ref        = np.datetime64('2011-09-10', 's')
llc_dt     = ref + llc_time.astype('timedelta64[s]')
llc_months = llc_dt.astype('datetime64[M]').astype(int) % 12 + 1   # 1=Jan ... 12=Dec

# Filter to March-July only (months 3-7), to match mooring season
spring_mask = np.isin(llc_months, [3, 4, 5, 6, 7])
print(f"LLC depth levels: {len(llc_depths)}  ({llc_depths[0]:.1f} - {llc_depths[-1]:.1f} m)")
print(f"LLC time steps total: {len(llc_time)}")
print(f"LLC March-July steps: {spring_mask.sum()}  (months found: {np.unique(llc_months[spring_mask])})")

# -- Collect per-station statistics --------------------------------------------
llc_means   = {}
llc_stds    = {}
moor_means  = {}
moor_stds   = {}
moor_depths = {}

print("\nLoading per-station data...")
for st in stations:
    # LLC data — filtered to March-July only
    llc_rho          = np.load(f'{llc_dir}{st}_llc_density.npy')
    llc_rho_filtered = llc_rho[spring_mask]
    llc_means[st]    = np.nanmean(llc_rho_filtered, axis=0)
    llc_stds[st]     = np.nanstd(llc_rho_filtered,  axis=0)

    # Mooring data
    moor_rho        = np.load(f'{moor_dir}{st}_density_interpolated.npy')
    moor_depths[st] = np.load(f'{moor_dir}{st}_depth.npy').flatten()
    moor_means[st]  = np.nanmean(moor_rho, axis=0)
    moor_stds[st]   = np.nanstd(moor_rho,  axis=0)

    print(f"  {st}: LLC filtered={llc_rho_filtered.shape}  "
          f"LLC mean={llc_means[st].mean():.2f} kg/m3  |  "
          f"Mooring shape={moor_rho.shape}  "
          f"Mooring mean={moor_means[st].mean():.2f} kg/m3")

# =============================================================================
# FIGURE 1 -- 11-panel grid, one panel per station
# =============================================================================
print("\nGenerating Figure 1: per-station panels...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharey=True)
axes_flat = axes.flatten()

for k, st in enumerate(stations):
    ax = axes_flat[k]

    lm = llc_means[st]
    ls = llc_stds[st]
    ax.plot(lm, -llc_depths, color='steelblue', lw=2, label='LLC4320 mean')
    ax.fill_betweenx(-llc_depths, lm - ls, lm + ls,
                     color='steelblue', alpha=0.2, label='LLC4320 +/-1 std')

    mm = moor_means[st]
    ms = moor_stds[st]
    md = moor_depths[st]
    ax.plot(mm, -md, color='darkorange', lw=2, label='Mooring mean')
    ax.fill_betweenx(-md, mm - ms, mm + ms,
                     color='darkorange', alpha=0.2, label='Mooring +/-1 std')

    ax.set_title(f'Station {st}', fontsize=11, fontweight='bold')
    ax.set_ylim(-195, 0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    if k % 4 == 0:
        ax.set_ylabel('Depth (m)', fontsize=9)
    if k >= 8:
        ax.set_xlabel('Density (kg/m3)', fontsize=9)

handles, labels = axes_flat[0].get_legend_handles_labels()
axes_flat[-1].axis('off')
fig.legend(handles, labels, loc='lower right', fontsize=10,
           bbox_to_anchor=(0.98, 0.05), framealpha=0.9)

fig.suptitle(
    'Raw Density Profiles: LLC4320 (Mar-Jul) vs Mooring Observations (Apr-Jun 2023)\n'
    'LLC filtered to March-July only to match mooring season  |  Shading = +/-1 std dev',
    fontsize=12, fontweight='bold'
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out1 = f'{out_dir}raw_comparison_panels.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved -> {out1}")

# =============================================================================
# FIGURE 2 -- All stations overlaid
# =============================================================================
print("\nGenerating Figure 2: all-station overlay...")

fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
colors = plt.cm.tab20(np.linspace(0, 1, len(stations)))

for k, st in enumerate(stations):
    c = colors[k]
    axes[0].plot(llc_means[st],  -llc_depths,     color=c, lw=1.8, label=st)
    axes[1].plot(moor_means[st], -moor_depths[st], color=c, lw=1.8, label=st)

for ax, title in zip(axes, ['LLC4320 (Mar-Jul only)',
                              'Moorings (April-June 2023)']):
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Density (kg/m3)', fontsize=11)
    ax.set_ylim(-195, 0)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('Depth (m)', fontsize=11)
axes[0].legend(fontsize=8, loc='lower left', ncol=2)

fig.suptitle(
    'Mean Raw Density -- All 11 Stations (seasonal filter applied to LLC)\n'
    'One color per station (same color in both panels)',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
out2 = f'{out_dir}raw_comparison_overlay.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved -> {out2}")

print("\n" + "=" * 60)
print("ALL DONE")
print(f"  {out1}")
print(f"  {out2}")
print("=" * 60)
