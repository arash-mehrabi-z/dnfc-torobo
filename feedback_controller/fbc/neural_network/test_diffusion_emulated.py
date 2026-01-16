"""
Example script for testing diffusion policy with emulated environment.
This script compares DNFC, Baseline, and Diffusion Policy performance.
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
use_diffusion = True
epoch_no = 10000  # Which checkpoint to load
train_num = 1   # Number of training runs to test
model_complexity = 'medium'  # 'low', 'medium', 'high', or 'xhigh'

# Create results directory
cur_file_dir_path = os.path.dirname(__file__)
results_dir = os.path.join(
    cur_file_dir_path,
    f'results/diffusion_emulated_{config.dataset_name}_{model_complexity}'
)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print(f"Results will be saved to: {results_dir}")

# Load models
print("\nLoading models...")
if use_diffusion:
    tester.load_diffusion_model(0, epoch_no, model_complexity)
    print("Loaded diffusion policy")
else:
    tester.load_model(0, epoch_no, config.use_custom_loss, model_complexity)
    print("Loaded DNFC and baseline models")

# Test on all episodes
print(f"\nTesting on {len(tester.dataset)} episodes...")

all_results = []
for eps_num in range(len(tester.dataset)):
    print(f"Episode {eps_num}/{len(tester.dataset)}")

    if use_diffusion:
        # Test diffusion policy
        y1_diff, y2_diff, y3_diff, y4_diff, y5_diff, y6_diff, y7_diff, path_point_diff = \
            tester.get_emulated_diffusion(eps_num, return_path_point=True)

        # # For comparison, also test baseline
        # y1_base, y2_base, y3_base, y4_base, y5_base, y6_base, y7_base, path_point_base = \
        #     tester.get_emulated(True, eps_num, return_path_point=True)

        success_diff = path_point_diff / 4.0
        # success_base = path_point_base / 4.0

        all_results.append({
            'eps_num': eps_num,
            'diffusion_success': success_diff,
            # 'baseline_success': success_base,
            'diffusion_path_point': path_point_diff,
            # 'baseline_path_point': path_point_base
        })

        print(f"  Diffusion: {path_point_diff}/4 milestones ({success_diff*100:.1f}%)")
        # print(f"  Baseline:  {path_point_base}/4 milestones ({success_base*100:.1f}%)")

    else:
        # Test DNFC vs Baseline
        y1_dnfc, y2_dnfc, y3_dnfc, y4_dnfc, y5_dnfc, y6_dnfc, y7_dnfc, path_point_dnfc = \
            tester.get_emulated(False, eps_num, return_path_point=True)

        y1_base, y2_base, y3_base, y4_base, y5_base, y6_base, y7_base, path_point_base = \
            tester.get_emulated(True, eps_num, return_path_point=True)

        success_dnfc = path_point_dnfc / 4.0
        success_base = path_point_base / 4.0

        all_results.append({
            'eps_num': eps_num,
            'dnfc_success': success_dnfc,
            'baseline_success': success_base,
            'dnfc_path_point': path_point_dnfc,
            'baseline_path_point': path_point_base
        })

        print(f"  DNFC:     {path_point_dnfc}/4 milestones ({success_dnfc*100:.1f}%)")
        print(f"  Baseline: {path_point_base}/4 milestones ({success_base*100:.1f}%)")

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

if use_diffusion:
    avg_diff = np.mean([r['diffusion_success'] for r in all_results])
    # avg_base = np.mean([r['baseline_success'] for r in all_results])
    print(f"Diffusion Policy: {avg_diff*100:.1f}% average success")
    # print(f"Baseline:         {avg_base*100:.1f}% average success")
else:
    avg_dnfc = np.mean([r['dnfc_success'] for r in all_results])
    avg_base = np.mean([r['baseline_success'] for r in all_results])
    print(f"DNFC:     {avg_dnfc*100:.1f}% average success")
    print(f"Baseline: {avg_base*100:.1f}% average success")

print(f"\nResults saved to: {results_file}")
