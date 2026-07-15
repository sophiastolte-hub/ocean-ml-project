"""
What script does:
- Loads the CNN model that was trained to predict subsurface density profiles from SSH patches
- Runs model on LLC4320 SSH patches collocated with mooring stations
- Reconstructs density profiles from predicted PCA coefficients
- Computes RMSE (error) at every depth level
- 2 Plots: RMSE vs depth, and predicted vs true example profiles

Data:
- LLC4320: global ocean simulation (not "real" data, but high-res simulation)
- The California Current subset covers Sept 2011-Nov 2012
- CNN input is 5x5 SSH patches, output is 3 PCA coefficients for density anomaly profiles
"""

import numpy as np
import torch
import torch.nn as nn
import sys
# imports the CNN class from D. Champenois's code repo
sys.path.insert(0, '/home/ds5324/ocean_ml_project/SWOT2Density')
from src.models import CNN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# All data is stored in /scratch/gpfs/LRGROUP/ds5324/
data_dir = '/scratch/gpfs/LRGROUP/ds5324/'

# Loads test data
# X_test: 5x5 SSH patches
# y_test: PCA coefficients 
print("Loading test data...")
X_test = np.load(f'{data_dir}/X_test.npy')
y_test = np.load(f'{data_dir}/y_test.npy')
pca_components = np.load(f'{data_dir}/pca_components.npy')[:3]
# density_mean: the average density anomaly profile across all training data
density_mean = np.load(f'{data_dir}/density_mean.npy').flatten()
# depths: depth levels in the LLC4320 simulation (negative values mean below seas surface)
depths = np.load(f'{data_dir}/depths_200m.npy')
print(f"Test set: {X_test.shape}")

# Load model
# Was having GPU problems, so now loading model on CPU and then moving to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CNN architecture: input is 5x5 SSH patch, output is 3 PCA coefficients, no auxiliary inputs
model = CNN(input_shape=(5,5,1), n_pca=3, aux_dim=0).to(device)
model.load_state_dict(torch.load(f'{data_dir}/model_llc4320.pt', map_location=device))
model.eval()

# Predict in batches
print("Running predictions...")
batch_size = 1024
all_preds = []
X_t = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i:i+batch_size].to(device)
        pred = model(batch, None)
        all_preds.append(pred.cpu().numpy())
y_pred = np.vstack(all_preds)

# Reconstruct density profiles
print("Reconstructing density profiles...")
density_pred = y_pred @ pca_components + density_mean
density_true = y_test @ pca_components + density_mean
# For the plots, we will subtract the mean density profile to show anomalies
density_pred_plot = density_pred - density_mean
density_true_plot  = density_true  - density_mean

# RMSE per depth level
rmse_per_depth = np.sqrt(np.mean((density_pred - density_true)**2, axis=0))
overall_rmse = np.sqrt(np.mean((density_pred - density_true)**2))
print(f"Overall RMSE: {overall_rmse:.6f} kg/m³")
print(f"Surface RMSE (top 50m): {rmse_per_depth[:5].mean():.6f} kg/m³")

# Plot 1: RMSE vs depth
# where in the water column is the model performing well or poorly?
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].plot(rmse_per_depth, depths, 'b-', linewidth=2)
axes[0].set_xlabel('RMSE (kg/m³)')
axes[0].set_ylabel('Depth (m)')
axes[0].set_title('RMSE vs Depth')
axes[0].grid(True)

# Plot 2: Example predicted vs true profiles (5 random)
# Solid lines - true (from simulation), dashed = CNN predictions
np.random.seed(42)
sample_idx = np.random.choice(len(y_test), 5)
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, idx in enumerate(sample_idx):
    axes[1].plot(density_true_plot[idx], depths, color=colors[i], linewidth=2, label=f'True {i+1}')
    axes[1].plot(density_pred_plot[idx], depths, color=colors[i], linewidth=2, linestyle='--', label=f'Pred {i+1}')
axes[1].set_xlabel('Density anomaly (kg/m³)')
axes[1].set_ylabel('Depth (m)')
axes[1].set_title('Predicted vs True Profiles')
axes[1].legend(fontsize=7)
axes[1].grid(True)

plt.tight_layout()
fig.suptitle('LLC4320 Test Set Evaluation', fontsize=13, fontweight='bold', y=1.02)
plt.savefig(f'{data_dir}/evaluation_results.png', dpi=150)
print(f"Plot saved!")
print("Done!")