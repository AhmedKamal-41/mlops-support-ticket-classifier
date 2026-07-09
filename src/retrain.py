"""
Retraining script for the support ticket classifier.

This script:
1. Loads data (can use updated dataset if available)
2. Trains a new model using the same pipeline as train.py
3. Compares the new model's performance with the current Production model
4. Automatically promotes the new model to Production if it performs better

Run this script manually when you want to retrain:
    python -m src.retrain
"""

import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_NAME,
    LABELS
)
from src.data_loader import load_support_ticket_data, train_val_test_split
from src.preprocessing import preprocess_texts, fit_vectorizer, transform_texts
from src.evaluation import evaluate_classification_model


def get_current_production_metrics() -> dict:
    """
    Get metrics from the current Production model.
    
    Returns:
        Dictionary with metrics, or None if no Production model exists.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        # Try to get Production model
        try:
            prod_versions = client.get_latest_versions(
                MODEL_NAME,
                stages=["Production"]
            )
            if not prod_versions:
                print("No Production model found.")
                return None
            
            prod_version = prod_versions[0]
            run_id = prod_version.run_id
            
            # Get metrics from the run
            run = client.get_run(run_id)
            metrics = run.data.metrics
            
            # Extract test F1 score (or validation F1 if test not available)
            if "test_weighted_f1" in metrics:
                f1_score = metrics["test_weighted_f1"]
            elif "test_macro_f1" in metrics:
                f1_score = metrics["test_macro_f1"]
            elif "val_weighted_f1" in metrics:
                f1_score = metrics["val_weighted_f1"]
            else:
                f1_score = None
            
            return {
                "version": prod_version.version,
                "f1_score": f1_score,
                "metrics": metrics
            }
            
        except (IndexError, Exception) as e:
            print(f"Could not retrieve Production model: {e}")
            return None
            
    except Exception as e:
        print(f"Error getting production metrics: {e}")
        return None


def promote_model_to_production(run_id: str, version: int):
    """
    Promote a model version to Production stage.
    
    Args:
        run_id: MLflow run ID.
        version: Model version number.
    """
    try:
        client = MlflowClient()
        
        # Transition model to Production
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production"
        )
        
        print(f"✓ Model version {version} promoted to Production stage")
        
        # Optionally archive previous Production versions
        # (This is optional - you might want to keep multiple Production versions)
        
    except Exception as e:
        print(f"✗ Error promoting model to Production: {e}")
        raise


def retrain_and_compare():
    """
    Main retraining function that trains a new model and compares it with Production.
    """
    print("=" * 60)
    print("Support Ticket Classifier - Retraining Pipeline")
    print("=" * 60)
    
    # Step 1: Get current Production model metrics
    print("\n[Step 1] Checking current Production model...")
    current_prod = get_current_production_metrics()
    
    if current_prod:
        print(f"  Current Production model:")
        print(f"    Version: {current_prod['version']}")
        if current_prod['f1_score']:
            print(f"    F1 Score: {current_prod['f1_score']:.4f}")
    else:
        print("  No Production model found. New model will be promoted automatically.")
    
    # Step 2: Train new model
    print("\n[Step 2] Training new model...")
    print("  (This will reuse the training pipeline from train.py)")
    
    # Load data
    df = load_support_ticket_data()
    train_df, val_df, test_df = train_val_test_split(df)
    
    # Preprocess
    train_texts = preprocess_texts(train_df['text'].tolist())
    val_texts = preprocess_texts(val_df['text'].tolist())
    test_texts = preprocess_texts(test_df['text'].tolist())
    
    # Create vectorizer
    vectorizer = fit_vectorizer(
        train_texts,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Transform
    X_train = transform_texts(vectorizer, train_texts)
    X_val = transform_texts(vectorizer, val_texts)
    X_test = transform_texts(vectorizer, test_texts)
    
    y_train = train_df['label'].tolist()
    y_val = val_df['label'].tolist()
    y_test = test_df['label'].tolist()
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Train model
    from sklearn.linear_model import LogisticRegression
    import joblib
    
    model_params = {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42,
        'solver': 'lbfgs',
        'multi_class': 'multinomial'
    }
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("vectorizer_max_features", 5000)
        mlflow.log_param("vectorizer_ngram_range", "(1, 2)")
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("val_size", len(val_df))
        mlflow.log_param("test_size", len(test_df))
        mlflow.log_param("retraining_run", True)
        
        # Train
        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_val_pred = model.predict(X_val)
        val_metrics = evaluate_classification_model(y_val, y_val_pred, labels=LABELS)
        
        for metric_name, metric_value in val_metrics.items():
            if isinstance(metric_value, float):
                mlflow.log_metric(f"val_{metric_name}", metric_value)
        
        y_test_pred = model.predict(X_test)
        test_metrics = evaluate_classification_model(y_test, y_test_pred, labels=LABELS)
        
        new_f1_score = test_metrics.get('weighted_f1', test_metrics.get('macro_f1'))
        
        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, float):
                mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME
        )
        
        # Log vectorizer
        vectorizer_path = "vectorizer.pkl"
        joblib.dump(vectorizer, vectorizer_path)
        mlflow.log_artifact(vectorizer_path, "vectorizer")
        if os.path.exists(vectorizer_path):
            os.remove(vectorizer_path)
        
        # Get the new model version
        run_id = mlflow.active_run().info.run_id
        client = MlflowClient()
        new_version = client.get_latest_versions(MODEL_NAME, stages=[])[-1]
        
        print(f"\n[Step 3] New model performance:")
        print(f"  Version: {new_version.version}")
        print(f"  Test F1 Score: {new_f1_score:.4f}")
        
        # Step 3: Compare and promote
        print("\n[Step 3] Comparing with Production model...")
        
        should_promote = False
        
        if current_prod is None:
            # No Production model exists, promote automatically
            print("  No Production model found. Promoting new model...")
            should_promote = True
        elif new_f1_score and current_prod['f1_score']:
            # Compare F1 scores
            improvement = new_f1_score - current_prod['f1_score']
            print(f"  F1 Score comparison:")
            print(f"    Current Production: {current_prod['f1_score']:.4f}")
            print(f"    New model: {new_f1_score:.4f}")
            print(f"    Improvement: {improvement:+.4f}")
            
            # Promote if new model is better (or equal)
            if improvement >= 0:
                print("  ✓ New model performs better or equal. Promoting to Production...")
                should_promote = True
            else:
                print("  ✗ New model performs worse. Keeping current Production model.")
                print(f"    (New model is saved but not promoted)")
        else:
            # Can't compare, but promote anyway (user can manually verify)
            print("  Could not compare metrics. Promoting new model...")
            should_promote = True
        
        if should_promote:
            promote_model_to_production(run_id, new_version.version)
        else:
            print(f"\n  To manually promote this model later, use MLflow UI or CLI:")
            print(f"    mlflow models transition-model-version-stage \\")
            print(f"      --name {MODEL_NAME} \\")
            print(f"      --version {new_version.version} \\")
            print(f"      --stage Production")
        
        print("\n" + "=" * 60)
        print("Retraining completed!")
        print("=" * 60)


if __name__ == "__main__":
    retrain_and_compare()

