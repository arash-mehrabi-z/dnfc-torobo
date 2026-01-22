"""
Example script for testing transformer-based diffusion policy with emulated environment.
This script tests the DiffusionTransformerPolicyModel performance.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from testers import Tester
from config import Config

config = Config()
tester = Tester()

# Test configuration
epoch_no = 4000  # Which checkpoint to load
train_num = 1   # Number of training runs to test
model_complexity = 'minimal'  # 'minimal', 'low', 'medium', 'high', or 'xhigh'

# Create results directory
cur_file_dir_path = os.path.dirname(__file__)
results_dir = os.path.join(
    cur_file_dir_path,
    f'results/diffusion_transformer_emulated_{config.dataset_name}_{model_complexity}'
)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print(f"Results will be saved to: {results_dir}")

# Load transformer diffusion model
print("\nLoading transformer diffusion model...")
tester.load_diffusion_transformer_model(0, epoch_no, model_complexity)
print("Loaded transformer diffusion policy")

# Test on all episodes
print(f"\nTesting on {len(tester.dataset)} episodes...")

all_results = []
for eps_num in range(len(tester.dataset)):
    print(f"Episode {eps_num}/{len(tester.dataset)}")

    # Test transformer diffusion policy
    y1_diff, y2_diff, y3_diff, y4_diff, y5_diff, y6_diff, y7_diff, path_point_diff = \
        tester.get_emulated_diffusion_transformer(eps_num, return_path_point=True)

    success_diff = path_point_diff / 4.0

    all_results.append({
        'eps_num': eps_num,
        'transformer_diffusion_success': success_diff,
        'transformer_diffusion_path_point': path_point_diff,
    })

    print(f"  Transformer Diffusion: {path_point_diff}/4 milestones ({success_diff*100:.1f}%)")

# Save results
results_file = os.path.join(results_dir, 'results.csv')
with open(results_file, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)

# Print summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)

avg_diff = np.mean([r['transformer_diffusion_success'] for r in all_results])
print(f"Transformer Diffusion Policy: {avg_diff*100:.1f}% average success")

print(f"\nResults saved to: {results_file}")
