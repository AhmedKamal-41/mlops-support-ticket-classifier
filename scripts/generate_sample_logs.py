"""
Generate sample inference logs for demonstration purposes.

This script creates realistic fake inference logs with:
- Timestamps spread over the last 24 hours (or configurable)
- Realistic latency values (50-500ms with some outliers)
- Status codes (mostly 200, some 500 errors)
- Predicted labels distributed across all categories
- Realistic patterns (more requests during business hours, error spikes)
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PROJECT_ROOT, LABELS


def generate_sample_logs(
    num_logs: int = 1000,
    output_path: str = None,
    hours_back: int = 24
):
    """
    Generate sample inference logs.
    
    Args:
        num_logs: Number of log entries to generate
        output_path: Output file path (default: logs/inference_logs.jsonl)
        hours_back: Number of hours back to spread timestamps
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "logs" / "inference_logs.jsonl"
    else:
        output_path = Path(output_path)
    
    # Create logs directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours_back)
    time_range_seconds = (end_time - start_time).total_seconds()
    
    # Generate logs
    logs = []
    
    # Define latency patterns by label (some labels might have higher latency)
    label_latency_base = {
        'billing': 80,
        'technical': 120,
        'account': 70,
        'shipping': 90,
        'general': 60
    }
    
    # Generate timestamps with more activity during business hours (9 AM - 5 PM)
    for i in range(num_logs):
        # Random timestamp
        random_seconds = random.uniform(0, time_range_seconds)
        timestamp = start_time + timedelta(seconds=random_seconds)
        
        # Bias towards business hours (9 AM - 5 PM)
        hour = timestamp.hour
        if 9 <= hour <= 17:
            # Higher probability during business hours
            if random.random() > 0.3:  # 70% chance to keep this timestamp
                pass
            else:
                # Re-roll to get more business hours
                random_seconds = random.uniform(0, time_range_seconds)
                timestamp = start_time + timedelta(seconds=random_seconds)
        
        # Select predicted label (weighted distribution)
        # Technical and billing are more common
        label_weights = {
            'billing': 0.25,
            'technical': 0.30,
            'account': 0.15,
            'shipping': 0.15,
            'general': 0.15
        }
        predicted_label = random.choices(
            list(label_weights.keys()),
            weights=list(label_weights.values())
        )[0]
        
        # Generate latency based on label and add some variance
        base_latency = label_latency_base[predicted_label]
        # Add random variance (50-200% of base)
        latency_multiplier = random.uniform(0.5, 2.0)
        latency_ms = base_latency * latency_multiplier
        
        # Add occasional outliers (very high latency)
        if random.random() < 0.05:  # 5% chance of outlier
            latency_ms = random.uniform(500, 2000)
        
        # Generate status code (mostly 200, some 500 errors)
        # Error spikes occasionally
        error_probability = 0.05  # 5% base error rate
        if random.random() < 0.1:  # 10% chance of error spike
            error_probability = 0.3  # 30% error rate during spike
        
        if random.random() < error_probability:
            status_code = 500
            # Errors might have higher latency
            latency_ms = latency_ms * random.uniform(1.5, 3.0)
        else:
            status_code = 200
        
        # Create log entry
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'latency_ms': round(latency_ms, 2),
            'status_code': status_code,
            'predicted_label': predicted_label
        }
        
        logs.append(log_entry)
    
    # Sort logs by timestamp
    logs.sort(key=lambda x: x['timestamp'])
    
    # Write to JSONL file
    with open(output_path, 'w') as f:
        for log_entry in logs:
            f.write(json.dumps(log_entry) + '\n')
    
    print(f"[OK] Generated {num_logs} sample inference logs")
    print(f"Saved to: {output_path}")
    print(f"Time range: {start_time.isoformat()} to {end_time.isoformat()}")
    
    # Print summary statistics
    status_200 = sum(1 for log in logs if log['status_code'] == 200)
    status_500 = sum(1 for log in logs if log['status_code'] == 500)
    avg_latency = sum(log['latency_ms'] for log in logs) / len(logs)
    
    print(f"\nSummary:")
    print(f"   - Status 200: {status_200} ({status_200/num_logs*100:.1f}%)")
    print(f"   - Status 500: {status_500} ({status_500/num_logs*100:.1f}%)")
    print(f"   - Average latency: {avg_latency:.2f} ms")
    
    label_counts = {}
    for log in logs:
        label = log['predicted_label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nLabel distribution:")
    for label in LABELS:
        count = label_counts.get(label, 0)
        print(f"   - {label}: {count} ({count/num_logs*100:.1f}%)")


def main():
    """CLI entrypoint for generating sample logs."""
    parser = argparse.ArgumentParser(
        description="Generate sample inference logs for demonstration"
    )
    parser.add_argument(
        '--num-logs',
        type=int,
        default=1000,
        help='Number of log entries to generate (default: 1000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: logs/inference_logs.jsonl)'
    )
    parser.add_argument(
        '--hours-back',
        type=int,
        default=24,
        help='Number of hours back to spread timestamps (default: 24)'
    )
    
    args = parser.parse_args()
    
    generate_sample_logs(
        num_logs=args.num_logs,
        output_path=args.output,
        hours_back=args.hours_back
    )


if __name__ == "__main__":
    main()

