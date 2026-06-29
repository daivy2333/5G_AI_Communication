"""
Collect real experimental data by running actual PyTorch training.
Uses the project's built-in trainer pipelines (not formula-generated curves).

Usage:
    /home/daivy/miniconda3/bin/python collect_experiment_data.py
    /home/daivy/miniconda3/bin/python collect_experiment_data.py --quick
    /home/daivy/miniconda3/bin/python collect_experiment_data.py --skip-scheduling

Output: docs/experiments/real/{data,figures,logs,tables,checkpoints}/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "docs" / "experiments" / "real"
for sub in ("data", "figures", "logs", "tables", "checkpoints"):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
})

C_BLUE = "#1F4E79"
C_RED = "#C0504D"
C_GREEN = "#9BBB59"
C_MID = "#4F81BD"


def export_csv(filename, headers, rows):
    path = OUT_DIR / "data" / filename
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)
    print(f"   CSV: {path}")


def export_log(filename, lines):
    path = OUT_DIR / "logs" / filename
    path.write_text("\n".join(lines) + "\n")
    print(f"   LOG: {path}")


def save_fig(filename, fig):
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / "figures" / f"{filename}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   FIG: {OUT_DIR / 'figures' / filename}.png")


def collect_channel_estimation(quick: bool):
    from channel_estimation.trainer import ChannelEstimationTrainer

    config = {
        "model_type": "transformer",
        "num_train_samples": 2000 if quick else 8000,
        "num_val_samples": 400 if quick else 2000,
        "epochs": 10 if quick else 100,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "snr_range": (-10, 30),
        "channel_model": "5G_UMa",
        "embedding_dim": 128,
        "num_heads": 8,
        "num_layers": 4,
        "dropout_rate": 0.1,
        "patience": 50,
    }
    print(f"  Config: samples={config['num_train_samples']}, epochs={config['epochs']}")
    trainer = ChannelEstimationTrainer(config)
    train_data, val_data = trainer.prepare_data()
    trainer.train_pytorch_model(train_data, val_data)

    history = trainer.train_history
    epochs = list(range(1, len(history["loss"]) + 1))

    test_dataset, test_labels = trainer.data_generator.generate_training_dataset(
        num_samples=1000, model=trainer.channel_model,
        snr_range=trainer.snr_range, seed=456)
    test_baselines = trainer.data_generator.compute_baseline_estimates(
        test_dataset["received_signal"], test_dataset["transmitted_pilot"])
    test_data = {
        "received": test_dataset["received_signal"],
        "pilot": test_dataset["transmitted_pilot"],
        "labels": test_labels,
        "snr": test_dataset["snr_db"],
        "baselines": test_baselines,
    }
    eval_results = trainer.evaluate(test_data)
    snr_results = trainer.evaluate_by_snr(test_data)

    export_csv("channel_estimation_loss.csv",
               ["epoch", "train_loss", "val_loss", "train_mse", "val_mse"],
               [[epochs[i], history["loss"][i], history["val_loss"][i],
                 history["mse"][i], history["val_mse"][i]] for i in range(len(epochs))])

    log_lines = [
        "[AI-ChannelNet] real training log",
        f"model=Transformer layers=4 heads=8 d_model=128 batch=64 lr=1e-4 epochs={len(epochs)}",
        f"seed=42 train_samples={config['num_train_samples']}",
    ]
    for i in range(len(epochs)):
        log_lines.append(
            f"epoch={epochs[i]:3d}  train_loss={history['loss'][i]:.6f}  "
            f"val_loss={history['val_loss'][i]:.6f}  "
            f"train_mse={history['mse'][i]:.6f}  val_mse={history['val_mse'][i]:.6f}")
    log_lines.append(f"[done] best_val_loss={min(history['val_loss']):.6f}")
    export_log("channel_estimation_train.log", log_lines)

    export_csv("channel_estimation_evaluation.csv",
               ["algorithm", "nmse_dB"],
               [[algo, round(val, 4)] for algo, val in eval_results.items()])

    snr_rows = []
    for snr_label, snr_vals in snr_results.items():
        for algo, val in snr_vals.items():
            snr_rows.append([snr_label, algo, round(val, 4)])
    export_csv("channel_estimation_by_snr.csv",
               ["snr_range", "algorithm", "nmse_dB"], snr_rows)

    trainer.save_model()

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.plot(epochs, history["loss"], label="Training loss", color=C_BLUE, lw=1.6)
    ax.plot(epochs, history["val_loss"], label="Validation loss", color=C_RED, lw=1.6, ls="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("AI-ChannelNet Training and Validation Loss")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    save_fig("real_channel_loss", fig)

    return {"best_val_loss": min(history["val_loss"]),
            "final_val_mse": history["val_mse"][-1],
            "evaluation": {k: round(float(v), 4) for k, v in eval_results.items()}}


def collect_signal_detection(quick: bool):
    from signal_detection.detector import DetectorTrainer

    n_samples = 1000 if quick else 5000
    config = {"epochs": 5 if quick else 50, "batch_size": 128, "learning_rate": 1e-3,
              "cnn_filters": [64, 128, 256], "lstm_units": 128, "dropout_rate": 0.3}
    print(f"  Config: samples={n_samples}, epochs={config['epochs']}")

    trainer = DetectorTrainer(config)
    trainer.train(num_samples=n_samples)
    results = trainer.evaluate(num_samples=max(200, n_samples // 4))

    hist = trainer.detector.detector.history
    epochs = list(range(1, len(hist["loss"]) + 1))
    has_val = len(hist.get("val_loss", [])) > 0

    export_csv("signal_detection_accuracy.csv",
               ["epoch", "train_loss", "val_loss", "train_acc_pct", "val_acc_pct"],
               [[epochs[i],
                 round(hist["loss"][i], 5),
                 round(hist["val_loss"][i], 5) if has_val and i < len(hist["val_loss"]) else "",
                 round(hist["accuracy"][i] * 100, 2),
                 round(hist["val_accuracy"][i] * 100, 2) if has_val and i < len(hist["val_accuracy"]) else ""]
                for i in range(len(epochs))])

    log_lines = [
        "[CNN-LSTM Detector] real training log",
        f"filters=64,128,256 lstm=128 batch=128 lr=1e-3 epochs={len(epochs)}",
        f"train_samples={n_samples}",
    ]
    for i in range(len(epochs)):
        val_info = ""
        if has_val and i < len(hist["val_loss"]):
            val_info = (f"val_loss={hist['val_loss'][i]:.4f}  "
                        f"val_acc={hist['val_accuracy'][i]*100:5.1f}%")
        log_lines.append(
            f"epoch={epochs[i]:3d}  loss={hist['loss'][i]:.4f}  "
            f"acc={hist['accuracy'][i]*100:5.1f}%  {val_info}")
    final_acc = max(hist["val_accuracy"]) * 100 if has_val else hist["accuracy"][-1] * 100
    log_lines.append(f"[done] best_val_acc={final_acc:.2f}%")
    export_log("signal_detection_train.log", log_lines)

    export_csv("signal_detection_results.csv",
               ["metric", "value"],
               [[k, round(float(v), 4)] for k, v in results.items()
                if k != "confusion_matrix"])

    val_acc_vals = [(hist["val_accuracy"][i] * 100) if has_val and i < len(hist["val_accuracy"])
                    else None for i in range(len(epochs))]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.plot(epochs, [a * 100 for a in hist["accuracy"]], label="Training acc", color=C_BLUE, lw=1.6)
    valid = [(epochs[i], v) for i, v in enumerate(val_acc_vals) if v is not None]
    if valid:
        ax.plot([e for e, _ in valid], [v for _, v in valid],
                label="Validation acc", color=C_MID, lw=1.6, ls="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("CNN-LSTM Detector Accuracy vs. Epoch")
    ax.set_ylim(0, 105); ax.legend(frameon=False); ax.grid(alpha=0.3)
    save_fig("real_detector_accuracy", fig)

    return {"final_val_acc_pct": round(final_acc, 2),
            "evaluation": {k: round(float(v), 4) for k, v in results.items()
                           if k != "confusion_matrix"}}


def collect_modulation_recognition(quick: bool):
    from signal_detection.recognizer import CNNModulationRecognizer
    from signal_detection.modulator import SignalGenerator

    n_per_class = 200 if quick else 3000
    n_extra_64qam = 0 if quick else 1500
    n_extra_256qam = 0 if quick else 3000
    n_classes = 5
    config = {"epochs": 5 if quick else 50, "batch_size": 128, "learning_rate": 1e-3,
              "dropout_rate": 0.4, "dense_units": 512}
    print(f"  Config: per_class={n_per_class}, FC=512, extra 64QAM={n_extra_64qam}, extra 256QAM={n_extra_256qam}")

    gen = SignalGenerator(config)
    data, labels = gen.generate_dataset(num_samples_per_class=n_per_class)

    if not quick and (n_extra_64qam > 0 or n_extra_256qam > 0):
        extra_x, extra_y = [], []
        mod_types = gen.modulation_types
        extras = {mod_types[3]: n_extra_64qam, mod_types[4]: n_extra_256qam}
        for mod_name, n_extra in extras.items():
            if n_extra <= 0:
                continue
            for _ in range(n_extra):
                snr_db = np.random.uniform(-10, 30)
                iq, label = gen.generate_iq_samples(mod_name, 128, snr_db)
                feat = np.zeros((gen.samples_per_symbol * 128, 2), dtype=np.float32)
                ml = min(len(iq), feat.shape[0])
                feat[:ml, 0] = iq.real[:ml].astype(np.float32)
                feat[:ml, 1] = iq.imag[:ml].astype(np.float32)
                extra_x.append(feat)
                extra_y.append(label)
        if extra_x:
            data = np.concatenate([data, np.array(extra_x)], axis=0)
            labels = np.concatenate([labels, np.array(extra_y)], axis=0)

    indices = np.random.permutation(len(data))
    split_idx = int(len(data) * 0.8)
    train_data, train_labels = data[indices[:split_idx]], labels[indices[:split_idx]]
    val_data, val_labels = data[indices[split_idx:]], labels[indices[split_idx:]]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    recognizer = CNNModulationRecognizer(n_classes, config)
    recognizer.train(train_data, train_labels, val_data, val_labels)
    results = recognizer.evaluate(
        *gen.generate_dataset(num_samples_per_class=max(100, n_per_class // 2)),
        by_snr=True,
        snr_values=np.random.uniform(-10, 30, max(100, n_per_class // 2) * n_classes))

    hist = recognizer.history
    epochs = list(range(1, len(hist.get("loss", [])) + 1))
    has_val = len(hist.get("val_loss", [])) > 0

    export_csv("modulation_recognition_accuracy.csv",
               ["epoch", "train_loss", "val_loss", "train_acc_pct", "val_acc_pct"],
               [[epochs[i],
                 round(hist["loss"][i], 5),
                 round(hist["val_loss"][i], 5) if has_val and i < len(hist["val_loss"]) else "",
                 round(hist["accuracy"][i] * 100, 2),
                 round(hist["val_accuracy"][i] * 100, 2) if has_val and i < len(hist["val_accuracy"]) else ""]
                for i in range(len(epochs))])

    log_lines = [
        "[CNN Mod Recognizer] real training log",
        f"classes=BPSK,QPSK,16QAM,64QAM,256QAM epochs={len(epochs)}",
        f"per_class={n_per_class}",
    ]
    for i in range(len(epochs)):
        val_info = ""
        if has_val and i < len(hist["val_loss"]):
            val_info = (f"val_loss={hist['val_loss'][i]:.4f}  "
                        f"val_acc={hist['val_accuracy'][i]*100:5.1f}%")
        log_lines.append(
            f"epoch={epochs[i]:3d}  loss={hist['loss'][i]:.4f}  "
            f"acc={hist['accuracy'][i]*100:5.1f}%  {val_info}")
    final_acc = max(hist["val_accuracy"]) * 100 if has_val else hist["accuracy"][-1] * 100
    log_lines.append(f"[done] best_val_acc={final_acc:.2f}%")
    export_log("modulation_recognition_train.log", log_lines)

    export_csv("modulation_recognition_results.csv",
               ["metric", "value"],
               [["overall_accuracy", round(float(results["overall_accuracy"]), 4)]] +
               [[f"class_{k}", round(float(v), 4)]
                for k, v in results.get("class_accuracy", {}).items()])

    if "accuracy_by_snr" in results:
        export_csv("modulation_recognition_by_snr.csv",
                   ["snr_range", "accuracy"],
                   [[snr, round(float(acc), 4)]
                    for snr, acc in results["accuracy_by_snr"].items()])

    val_acc_vals = [(hist["val_accuracy"][i] * 100) if has_val and i < len(hist["val_accuracy"])
                    else None for i in range(len(epochs))]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.plot(epochs, [a * 100 for a in hist["accuracy"]], label="Training acc", color=C_BLUE, lw=1.6)
    valid = [(epochs[i], v) for i, v in enumerate(val_acc_vals) if v is not None]
    if valid:
        ax.plot([e for e, _ in valid], [v for _, v in valid],
                label="Validation acc", color=C_MID, lw=1.6, ls="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Modulation Recognizer Accuracy vs. Epoch")
    ax.set_ylim(0, 105); ax.legend(frameon=False); ax.grid(alpha=0.3)
    save_fig("real_modulation_accuracy", fig)

    return {"final_val_acc_pct": round(final_acc, 2),
            "evaluation": {k: round(float(v), 4) for k, v in results.items()
                           if k not in ("class_accuracy", "accuracy_by_snr",
                                        "predictions", "true_labels", "probabilities")}}


def collect_resource_scheduling(quick: bool):
    from resource_scheduling.agent import PPOScheduler

    timesteps = 2000 if quick else 10000
    env_config = {"num_users": 10 if quick else 20,
                  "num_resource_blocks": 50 if quick else 100, "time_slots": 10}
    agent_config = {"gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
                    "entropy_coef": 0.01, "learning_rate": 3e-4, "hidden_dim": 64}
    print(f"  Config: users={env_config['num_users']}, RBs={env_config['num_resource_blocks']}, "
          f"timesteps={timesteps}")

    scheduler = PPOScheduler(env_config, agent_config)
    log_interval = max(1, timesteps // 50)
    scheduler.train(timesteps, log_interval=log_interval)

    stats = scheduler.training_stats
    n_ep = len(stats["episode_rewards"])
    episodes = list(range(1, n_ep + 1))
    rewards = [float(r) for r in stats["episode_rewards"]]
    throughputs = [float(t) for t in stats["throughputs"]]
    fairness_vals = [float(f) for f in stats["fairness"]]

    scheduler.save(str(OUT_DIR / "checkpoints" / "ppo_scheduler"))

    export_csv("resource_scheduling_reward.csv",
               ["episode", "reward", "throughput_Mbps", "fairness"],
               [[episodes[i], round(rewards[i], 4), round(throughputs[i], 2), round(fairness_vals[i], 4)]
                for i in range(n_ep)])

    log_lines = [
        "[PPO Scheduler] real training log",
        f"gamma=0.99 gae_lambda=0.95 clip=0.2 ent_coef=0.01 lr=3e-4",
        f"users={env_config['num_users']} rbs={env_config['num_resource_blocks']} timesteps={timesteps}",
    ]
    for i in range(n_ep):
        log_lines.append(
            f"episode={episodes[i]:3d}  reward={rewards[i]:+.4f}  "
            f"throughput={throughputs[i]:6.2f}Mbps  fairness={fairness_vals[i]:.4f}")
    avg_r = float(np.mean(rewards[-20:])) if n_ep >= 20 else float(np.mean(rewards))
    log_lines.append(
        f"[done] avg_reward(last20)={avg_r:.4f}  "
        f"final_throughput={throughputs[-1]:.2f}Mbps  final_fairness={fairness_vals[-1]:.4f}")
    export_log("resource_scheduling_train.log", log_lines)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.plot(episodes, rewards, color=C_BLUE, alpha=0.3, lw=0.8, label="Raw reward")
    if n_ep >= 10:
        smooth = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        ax.plot(episodes[9:], smooth, color=C_BLUE, lw=1.8, label="Smoothed (10-ep MA)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.set_title("PPO Scheduler Convergence")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    save_fig("real_ppo_reward", fig)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.plot(episodes, throughputs, color=C_BLUE, lw=1.2)
    ax.set_xlabel("Episode"); ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("PPO Scheduler Throughput")
    ax.grid(alpha=0.3)
    save_fig("real_ppo_throughput", fig)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.plot(episodes, fairness_vals, color=C_GREEN, lw=1.2)
    ax.set_xlabel("Episode"); ax.set_ylabel("Jain Fairness Index")
    ax.set_title("PPO Scheduler Fairness")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    save_fig("real_ppo_fairness", fig)

    return {"avg_reward_last20": round(avg_r, 4),
            "final_throughput_Mbps": round(throughputs[-1], 2),
            "final_fairness": round(fairness_vals[-1], 4)}


def main():
    parser = argparse.ArgumentParser(description="Collect real experimental data")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-channel", action="store_true")
    parser.add_argument("--skip-detection", action="store_true")
    parser.add_argument("--skip-modulation", action="store_true")
    parser.add_argument("--skip-scheduling", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"Real Experiment Data Collection — {datetime.now().isoformat()}")
    print(f"Device: {device}  |  Quick: {args.quick}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    summary = {"timestamp": datetime.now().isoformat(), "device": str(device), "modules": {}}

    if not args.skip_channel:
        print("\n── Channel Estimation (AI-ChannelNet) ──")
        t0 = time.time()
        summary["modules"]["channel_estimation"] = collect_channel_estimation(args.quick)
        print(f"  Done in {time.time() - t0:.1f}s")

    if not args.skip_detection:
        print("\n── Signal Detection (CNN-LSTM) ──")
        t0 = time.time()
        summary["modules"]["signal_detection"] = collect_signal_detection(args.quick)
        print(f"  Done in {time.time() - t0:.1f}s")

    if not args.skip_modulation:
        print("\n── Modulation Recognition (CNN) ──")
        t0 = time.time()
        summary["modules"]["modulation_recognition"] = collect_modulation_recognition(args.quick)
        print(f"  Done in {time.time() - t0:.1f}s")

    if not args.skip_scheduling:
        print("\n── Resource Scheduling (PPO) ──")
        t0 = time.time()
        summary["modules"]["resource_scheduling"] = collect_resource_scheduling(args.quick)
        print(f"  Done in {time.time() - t0:.1f}s")

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n{'=' * 60}")
    print(f"All done. Output: {OUT_DIR}")
    print(f"Summary: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
