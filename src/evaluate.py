# src/evaluate.py
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model,
                   test_loader,
                   device='cuda',
                   model_name='model'):
    """
    Full evaluation with all metrics.

    Args:
        model:       Trained PyTorch model.
        test_loader: Test DataLoader.
        device:      'cuda' or 'cpu' or torch.device.
        model_name:  Name for reporting.

    Returns:
        metrics: Dictionary of all metrics.
    """
    model.eval()
    model = model.to(device)

    all_labels        = []
    all_predictions   = []
    all_probabilities = []
    latencies         = []

    # Handle both string 'cuda' and torch.device('cuda')
    device_type = device if isinstance(device, str) else device.type

    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(device)
            # frames shape: (batch, seq_len, C, H, W) always

            # --- Latency timing ---
            if device_type == 'cuda':
                torch.cuda.synchronize()

            start   = time.perf_counter()
            outputs = model(frames)         # shape [batch, 2]

            if device_type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()

            # Per-sample latency in ms (divide by batch size, not seq_len)
            batch_size = frames.shape[0]
            latency    = (end - start) / batch_size * 1000
            latencies.append(latency)

            # --- Predictions ---
            # softmax -> probability of FAKE class (class index 1)
            # shape: [batch]  (one probability per sample)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            # argmax -> predicted class index, shape: [batch]
            _, predicted = torch.max(outputs, 1)
            preds        = predicted.cpu().numpy()

            # labels are float32 from dataset -> convert to int for sklearn
            all_labels.extend(labels.long().numpy())
            all_predictions.extend(preds)           # [batch] ints
            all_probabilities.extend(probs)         # [batch] floats

    # --- Metrics ---
    metrics = {
        'model_name':       model_name,
        'accuracy':         accuracy_score(all_labels, all_predictions),
        'f1_score':         f1_score(all_labels, all_predictions,
                                     average='weighted'),
        'precision':        precision_score(all_labels, all_predictions,
                                            average='weighted'),
        'recall':           recall_score(all_labels, all_predictions,
                                         average='weighted'),
        'auc':              roc_auc_score(all_labels, all_probabilities),
        'avg_latency_ms':   float(np.mean(latencies)),
        'max_latency_ms':   float(np.max(latencies)),
        'min_latency_ms':   float(np.min(latencies)),
        'real_time_capable': bool(np.mean(latencies) < 200)
    }

    # --- Print results ---
    print("\n" + "=" * 55)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 55)
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"F1 Score:          {metrics['f1_score']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"AUC:               {metrics['auc']:.4f}")
    print(f"Avg Latency (ms):  {metrics['avg_latency_ms']:.2f}")
    print(f"Real-time capable: {metrics['real_time_capable']}")
    print("=" * 55)

    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_predictions,
        target_names=['Real', 'Fake']
    ))

    return metrics


def compare_models(cnn_metrics, lstm_metrics):
    """Compare CNN baseline vs CNN+LSTM."""
    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)

    metrics_to_compare = [
        'accuracy',
        'f1_score',
        'precision',
        'recall',
        'auc',
        'avg_latency_ms'
    ]

    print(f"{'Metric':<20} {'CNN':>10} "
          f"{'CNN+LSTM':>10} {'Improvement':>12}")
    print("-" * 55)

    for metric in metrics_to_compare:
        cnn_val  = cnn_metrics[metric]
        lstm_val = lstm_metrics[metric]

        if metric == 'avg_latency_ms':
            diff   = cnn_val - lstm_val
            symbol = "faster" if diff > 0 else "slower"
            print(f"{metric:<20} {cnn_val:>10.4f} "
                  f"{lstm_val:>10.4f} "
                  f"{abs(diff):>8.4f} {symbol}")
        else:
            diff   = lstm_val - cnn_val
            symbol = "better" if diff > 0 else "worse"
            print(f"{metric:<20} {cnn_val:>10.4f} "
                  f"{lstm_val:>10.4f} "
                  f"{abs(diff):>8.4f} {symbol}")

    print("=" * 55)

    # Hypothesis validation
    # F1 absolute improvement >= 0.05 (5 percentage points)
    f1_improvement_pct = (
        lstm_metrics['f1_score'] - cnn_metrics['f1_score']
    ) * 100
    latency_ok = lstm_metrics['avg_latency_ms'] < 200

    print("\nHYPOTHESIS VALIDATION:")
    print(f"F1 improvement:  {f1_improvement_pct:+.2f} percentage points")
    print(f"Target:          >= 5 percentage points")
    print(f"F1 hypothesis:   "
          f"{'VALIDATED' if f1_improvement_pct >= 5 else 'NOT VALIDATED'}")
    print(f"Latency < 200ms: "
          f"{'VALIDATED' if latency_ok else 'NOT VALIDATED'} "
          f"({lstm_metrics['avg_latency_ms']:.1f}ms)")

    if f1_improvement_pct >= 5 and latency_ok:
        print("\nOVERALL HYPOTHESIS: VALIDATED")
    else:
        print("\nOVERALL HYPOTHESIS: NOT VALIDATED")


def plot_results(cnn_history,
                 lstm_history,
                 save_dir='results/'):
    """Plot training curves for both models."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'CNN Baseline vs CNN+LSTM Training Results',
        fontsize=14
    )

    # CNN Loss
    axes[0, 0].plot(cnn_history['train_loss'],
                    label='Train', color='blue')
    axes[0, 0].plot(cnn_history['val_loss'],
                    label='Val',   color='orange')
    axes[0, 0].set_title('CNN Baseline - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # CNN+LSTM Loss
    axes[0, 1].plot(lstm_history['train_loss'],
                    label='Train', color='blue')
    axes[0, 1].plot(lstm_history['val_loss'],
                    label='Val',   color='orange')
    axes[0, 1].set_title('CNN+LSTM - Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # CNN Accuracy
    axes[1, 0].plot(cnn_history['train_acc'],
                    label='Train', color='green')
    axes[1, 0].plot(cnn_history['val_acc'],
                    label='Val',   color='red')
    axes[1, 0].set_title('CNN Baseline - Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # CNN+LSTM Accuracy
    axes[1, 1].plot(lstm_history['train_acc'],
                    label='Train', color='green')
    axes[1, 1].plot(lstm_history['val_acc'],
                    label='Val',   color='red')
    axes[1, 1].set_title('CNN+LSTM - Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'training_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Results saved to: {save_path}")