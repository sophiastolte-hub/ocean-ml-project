"""
mooring_eval.py
---------------
Evaluate the LLC4320-trained CNN on SWOT SSH patches collocated with
California Current mooring stations.

What this script does:
  - Takes the CNN that was trained on simulated LLC4320 SSH data.
  - Feeds it REAL SWOT satellite SSH patches collocated with mooring stations.
  - Compares the CNN's predicted density profiles to density actually MEASURED
    by the mooring instruments.
  - This tests whether the model generalizes from simulation to reality.

Data context:
  - Mooring stations: 11 stations (S1-S4, P1-P7) in the California Current
  - Mooring data period: April - June 2023
  - 1550 matched SSH patch + mooring profile pairs
  - Mooring depth grid: 0-195 m in 5 m steps (40 depth levels)
  - LLC4320 model depth grid: 28 levels (interpolated to match moorings)

Key finding:
  - The CNN performs well on LLC4320 test data (RMSE ~0.96 kg/m³)
  - But performance drops significantly on real mooring data (RMSE ~2.81 kg/m³)
  - This "domain gap" is likely due to differences between simulated and real SSH
    statistics, and different PCA bases for the two datasets.

"""

import numpy as np
import torch
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the CNN architecture from Dr. Champenois's SWOT2Density repository
sys.path.insert(0, '/home/ds5324/ocean_ml_project/SWOT2Density/src')
from models import CNN

ds_dir  = '/scratch/gpfs/LRGROUP/ds5324/'
# Dr. Champenois's preprocessed mooring + SWOT SSH data
bc_data = '/scratch/gpfs/LRGROUP/bc5848/california_current/data/processed/'

# This model was trained on LLC4320 simulation data.
# Architecture: 5x5 SSH patch -> 3 PCA coefficients -> 28-depth density profile
print("=" * 60)
print("Loading LLC4320 CNN model...")
model = CNN(input_shape=(5, 5, 1), n_pca=3, aux_dim=0)
model.load_state_dict(torch.load(f'{ds_dir}model_llc4320.pt', map_location='cpu'))
model.eval()
print("  Model loaded OK")

# These statistics describe the LLC4320 training data, NOT the real ocean.
# Using them to normalize real SWOT SSH is maybe a source of the domain gap.
print("\nLoading LLC4320 PCA and normalization...")
# np.abs becayse stored as negatives
depths_llc   = np.abs(np.load(f'{ds_dir}depths_200m.npy'))

# Mean density anomaly profile across all LLC4320 training samples
# Added back during reconstruction: density = PCA_coeff @ PCA_modes + density_mean
density_mean = np.load(f'{ds_dir}density_mean.npy').flatten()
pca_comps    = np.load(f'{ds_dir}pca_components.npy')[:3]
patch_mean   = np.load(f'{ds_dir}patch_mean.npy').squeeze()
patch_std    = np.load(f'{ds_dir}patch_std.npy').squeeze()

# sorts depths from shallow to deep
sort_idx     = np.argsort(depths_llc)
depths_llc   = depths_llc[sort_idx]
density_mean = density_mean[sort_idx]
pca_comps    = pca_comps[:, sort_idx]
print(f"  depths: {depths_llc[:4]} ...")

print("\nLoading SWOT SSH patches...")
ssh_patches = np.load(f'{bc_data}ML/ssh_patches_5x5.npy')
print(f"  Shape: {ssh_patches.shape}, range: {ssh_patches.min():.3f} to {ssh_patches.max():.3f}")

print("\nLoading mooring PCA ground truth...")
mooring_depths = np.arange(0, 200, 5)
pca_mean_moor  = np.load(f'{bc_data}pca_anomaly/pca_density_mean.npy').flatten()
pca_modes_moor = np.load(f'{bc_data}pca_anomaly/pca_modes.npy')
pca_targets    = np.load(f'{bc_data}ML/pca_anom_targets.npy')
true_anom      = pca_targets @ pca_modes_moor + pca_mean_moor
print(f"  True anomaly shape: {true_anom.shape}")

print("\nNormalizing and running CNN...")
patches_2d   = ssh_patches[:, :, :, 0]
patches_norm = (patches_2d - patch_mean) / (patch_std + 1e-8)
patches_4d   = patches_norm[:, :, :, np.newaxis]

X = torch.tensor(patches_4d, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X).numpy()
print(f"  Predicted coeffs shape: {y_pred.shape}")

pred_anom_llc = density_mean + (y_pred @ pca_comps)

print("Interpolating to mooring depth grid...")
pred_anom_40 = np.zeros((len(pred_anom_llc), 40))
for i in range(len(pred_anom_llc)):
    pred_anom_40[i] = np.interp(mooring_depths, depths_llc, pred_anom_llc[i])

print("\n" + "=" * 60)

# Per-sample RMSE, overall RMSE, RMSE by depth, and baseline RMSE
rmse_per_sample = np.sqrt(np.nanmean((pred_anom_40 - true_anom) ** 2, axis=1))
overall_rmse    = np.nanmean(rmse_per_sample)
rmse_by_depth   = np.sqrt(np.nanmean((pred_anom_40 - true_anom) ** 2, axis=0))
baseline_rmse   = np.sqrt(np.nanmean(true_anom ** 2))

print(f"Overall RMSE:    {overall_rmse:.4f} kg/m³")
print(f"Baseline RMSE:   {baseline_rmse:.4f} kg/m³")
print(f"Skill score:     {1 - overall_rmse/baseline_rmse:.4f}")
print(f"Surface RMSE:    {np.nanmean(rmse_by_depth[:10]):.4f} kg/m³")
print(f"Mid-depth RMSE:  {np.nanmean(rmse_by_depth[10:30]):.4f} kg/m³")

np.save(f'{ds_dir}mooring_pred_anom.npy',     pred_anom_40.astype(np.float32))
np.save(f'{ds_dir}mooring_true_anom.npy',     true_anom.astype(np.float32))
np.save(f'{ds_dir}mooring_rmse_by_depth.npy', rmse_by_depth.astype(np.float32))

print("\nGenerating plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('LLC4320 CNN Evaluated on Mooring Data', fontsize=13, fontweight='bold')

# Plot 1: RMSE vs depth
ax = axes[0]
ax.plot(rmse_by_depth, -mooring_depths, 'steelblue', linewidth=2.5)
ax.axvline(overall_rmse, color='red', linestyle='--', linewidth=1.5, label=f'Overall = {overall_rmse:.3f}')
ax.axvline(baseline_rmse, color='gray', linestyle=':', linewidth=1.5, label=f'Baseline = {baseline_rmse:.3f}')
ax.set_xlabel('RMSE (kg/m³)'); ax.set_ylabel('Depth (m)'); ax.set_title('RMSE vs Depth')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-195, 0)

# Plot 2: Example predicted vs true profiles (6 random)
ax = axes[1]
for k in range(6):
    c = plt.cm.tab10(k / 10)
    ax.plot(true_anom[k*250],    -mooring_depths, color=c, lw=2)
    ax.plot(pred_anom_40[k*250], -mooring_depths, color=c, lw=2, linestyle='--')
ax.plot([], [], 'k-',  lw=2, label='True'); ax.plot([], [], 'k--', lw=2, label='Predicted')
ax.set_xlabel('Density Anomaly (kg/m³)'); ax.set_ylabel('Depth (m)')
ax.set_title('Example Profiles'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-195, 0)

# Plot 3: True vs Predicted Anomaly Scatter
ax = axes[2]
t_flat = true_anom[::3].flatten(); p_flat = pred_anom_40[::3].flatten()
ax.scatter(t_flat, p_flat, alpha=0.15, s=3, color='steelblue', rasterized=True)
lim = np.nanpercentile(np.abs(np.concatenate([t_flat, p_flat])), 99) * 1.1
ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1.5, label='1:1')
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('True Anomaly (kg/m³)'); ax.set_ylabel('Predicted Anomaly (kg/m³)')
ax.set_title(f'True vs Predicted\nRMSE={overall_rmse:.3f}'); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{ds_dir}mooring_evaluation.png', dpi=150, bbox_inches='tight')
print(f"Plot saved.")
print("\nALL DONE")