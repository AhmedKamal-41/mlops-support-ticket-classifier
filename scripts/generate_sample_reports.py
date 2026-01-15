"""
Generate sample classification reports and confusion matrices for demonstration.

This script creates:
- reports/classification_report.json - Classification metrics
- reports/confusion_matrix.png - Confusion matrix visualization
- reports/drift_report.json - Optional drift detection report
"""

import argparse
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PROJECT_ROOT, LABELS

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/seaborn not available. Confusion matrix will not be generated.")


def generate_classification_report(labels, output_path: Path):
    """Generate a sample classification report JSON."""
    
    # Generate realistic metrics
    accuracy = round(random.uniform(0.82, 0.90), 4)
    
    metrics = {
        "accuracy": accuracy,
        "macro_precision": round(random.uniform(0.80, 0.88), 4),
        "macro_recall": round(random.uniform(0.80, 0.88), 4),
        "macro_f1": round(random.uniform(0.80, 0.88), 4),
        "weighted_precision": round(random.uniform(0.82, 0.89), 4),
        "weighted_recall": accuracy,
        "weighted_f1": round(random.uniform(0.82, 0.89), 4),
    }
    
    per_class = {}
    
    # Generate per-class metrics
    for label in labels:
        precision = round(random.uniform(0.70, 0.95), 4)
        recall = round(random.uniform(0.70, 0.95), 4)
        f1 = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        support = random.randint(5, 15)
        
        metrics[f"{label}_precision"] = precision
        metrics[f"{label}_recall"] = recall
        metrics[f"{label}_f1"] = f1
        metrics[f"{label}_support"] = support
        
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }
    
    report = {
        "metrics": metrics,
        "summary": {
            "accuracy": accuracy,
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"]
        },
        "per_class": per_class
    }
    
    # Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[OK] Generated classification report: {output_path}")
    return report


def generate_confusion_matrix(labels, output_path: Path, report: dict):
    """Generate a confusion matrix PNG image."""
    
    if not HAS_MATPLOTLIB:
        print("[WARNING] Skipping confusion matrix (matplotlib not available)")
        return
    
    n_labels = len(labels)
    
    # Generate confusion matrix values
    # Most predictions should be on diagonal (correct)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j:
                # Diagonal: correct predictions
                support = report["per_class"][true_label]["support"]
                # 80-95% correct predictions
                cm[i, j] = int(support * random.uniform(0.80, 0.95))
            else:
                # Off-diagonal: incorrect predictions
                # Some small random errors
                cm[i, j] = random.randint(0, 2)
    
    # Normalize rows so they sum to support
    for i, label in enumerate(labels):
        current_sum = cm[i, :].sum()
        if current_sum > 0:
            support = report["per_class"][label]["support"]
            scale = support / current_sum
            cm[i, :] = (cm[i, :] * scale).astype(int)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Use seaborn for better visualization
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Generated confusion matrix: {output_path}")


def generate_drift_report(output_path: Path):
    """Generate a sample drift detection report."""
    
    drift_score = round(random.uniform(0.2, 1.2), 3)
    
    # Determine status
    if drift_score < 0.5:
        status = "healthy"
    elif drift_score < 1.0:
        status = "degraded"
    else:
        status = "significant"
    
    report = {
        "drift_score": drift_score,
        "status": status,
        "reference_stats": {
            "mean": round(random.uniform(40, 50), 1),
            "std": round(random.uniform(10, 15), 1),
            "min": random.randint(5, 10),
            "max": random.randint(80, 100)
        },
        "recent_stats": {
            "mean": round(random.uniform(42, 52), 1),
            "std": round(random.uniform(11, 16), 1),
            "min": random.randint(5, 12),
            "max": random.randint(85, 105)
        },
        "timestamp": datetime.now().isoformat(),
        "sample_size": random.randint(500, 1000)
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[OK] Generated drift report: {output_path}")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate sample classification reports and confusion matrices"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: reports/)'
    )
    parser.add_argument(
        '--include-drift',
        action='store_true',
        help='Also generate drift report'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "reports"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate classification report
    report_path = output_dir / "classification_report.json"
    report = generate_classification_report(LABELS, report_path)
    
    # Generate confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    generate_confusion_matrix(LABELS, cm_path, report)
    
    # Generate drift report if requested
    if args.include_drift:
        drift_path = output_dir / "drift_report.json"
        generate_drift_report(drift_path)
    
    print(f"\n[OK] Sample reports generated in: {output_dir}")


if __name__ == "__main__":
    main()

