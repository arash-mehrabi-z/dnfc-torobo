"""Mean/std of score and DTW for the image GeneralModel across latent dims {4, 7, 14}.

Replicates the methodology in get_results.ipynb (cells 23-24): for each model's
perf.csv we tag every episode with its train round, average over episodes within
each round, then report the mean and std *across* the train rounds. This treats
each independently trained seed (train round) as one sample, which is what the
paper's error bars are built from.

perf.csv columns (written by online_tester.py for the image model):
    eps_num, image_model_succ (score), image_model_dtw (DTW), image_model_norm
"""
import os
import math
import pandas as pd

root_dir = "/home/arash/catkin_ws/src/feedback_controller/fbc/neural_network/results/"

# latent_dim -> perf.csv path (ep:5000, DT=0.1). 14 uses the suffix-less v_name.
result_files = {
    4:  "trajs:360_blocks:3_imgs_2cams_898.711K_interp_0.85/ep:5000/"
        "image_model_img_task_enc_rob_enc_raw_action_lat4DT=0.1/perf.csv",
    7:  "trajs:360_blocks:3_imgs_2cams_902.005K_interp_0.85/ep:5000/"
        "image_model_img_task_enc_rob_enc_raw_action_lat7DT=0.1/perf.csv",
    14: "trajs:360_blocks:3_imgs_2cams_909.691K_interp_0.85/ep:5000/"
        "image_model_img_task_enc_rob_enc_raw_actionDT=0.1/perf.csv",
}

SCORE_COL = "image_model_succ"
DTW_COL = "image_model_dtw"


def get_mean_std_across_trains(df):
    """Average over episodes per train round, then mean/std over rounds."""
    # train_round: 1, 2, 3, ... for the repeated runs of the same episode.
    df = df.copy()
    df["train_round"] = df.groupby("eps_num").cumcount() + 1
    round_means = df.groupby("train_round")[[SCORE_COL, DTW_COL]].mean()
    n = len(round_means)  # number of train rounds (seeds)
    overall_mean = round_means.mean()
    overall_std = round_means.std()  # pandas std is sample (ddof=1)
    return overall_mean, overall_std, n


rows = []
for latent_dim in sorted(result_files):
    path = os.path.join(root_dir, result_files[latent_dim])
    df = pd.read_csv(path)
    overall_mean, overall_std, n = get_mean_std_across_trains(df)

    score_mean, score_std = overall_mean[SCORE_COL], overall_std[SCORE_COL]
    dtw_mean, dtw_std = overall_mean[DTW_COL], overall_std[DTW_COL]
    # Standard error of the mean across train rounds (used for paper error bars).
    score_sem = score_std / math.sqrt(n)
    dtw_sem = dtw_std / math.sqrt(n)

    rows.append({
        "latent_dim": latent_dim,
        "n_rounds": n,
        "score_mean": score_mean,
        "score_std": score_std,
        "score_sem": score_sem,
        "dtw_mean": dtw_mean,
        "dtw_std": dtw_std,
        "dtw_sem": dtw_sem,
    })

results = pd.DataFrame(rows).set_index("latent_dim")
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
pd.set_option("display.width", 120)

print("Per-train-round aggregation (each seed = 1 sample), ep:5000, DT=0.1\n")
print(results)

print("\nFormatted (mean +/- std):")
for latent_dim, r in results.iterrows():
    print(f"  latent_dim={latent_dim:>2}  "
          f"score = {r['score_mean']:.3f} +/- {r['score_std']:.3f}    "
          f"DTW = {r['dtw_mean']:.2f} +/- {r['dtw_std']:.2f}    "
          f"(n={int(r['n_rounds'])})")

out_path = os.path.join(os.path.dirname(__file__),
                        "results_latent_score_dtw.csv")
results.to_csv(out_path)
print(f"\nSaved -> {out_path}")
