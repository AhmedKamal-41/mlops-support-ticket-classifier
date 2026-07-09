"""
Training script for the support ticket classifier.

This script:
1. Loads and preprocesses the support ticket dataset
2. Trains a LogisticRegression classifier
3. Evaluates the model on validation and test sets
4. Logs everything to MLflow (parameters, metrics, model, artifacts)
5. Registers the model in MLflow Model Registry

Run this script with:
    python -m src.train
"""

import os
import sys
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

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
from src.evaluation import evaluate_classification_model, print_classification_report


def train_model():
    """
    Main training function that orchestrates the entire training pipeline.
    """
    print("=" * 60)
    print("Support Ticket Classifier - Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load the dataset
    print("\n[Step 1] Loading dataset...")
    df = load_support_ticket_data()
    
    # Step 2: Split into train/val/test sets
    print("\n[Step 2] Splitting dataset...")
    train_df, val_df, test_df = train_val_test_split(df)
    
    # Step 3: Preprocess text data
    print("\n[Step 3] Preprocessing text data...")
    train_texts = preprocess_texts(train_df['text'].tolist())
    val_texts = preprocess_texts(val_df['text'].tolist())
    test_texts = preprocess_texts(test_df['text'].tolist())
    
    # Step 4: Create and fit the TF-IDF vectorizer
    print("\n[Step 4] Creating TF-IDF vectorizer...")
    vectorizer = fit_vectorizer(
        train_texts,
        max_features=5000,
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )
    
    # Step 5: Transform texts to feature vectors
    print("\n[Step 5] Transforming texts to feature vectors...")
    X_train = transform_texts(vectorizer, train_texts)
    X_val = transform_texts(vectorizer, val_texts)
    X_test = transform_texts(vectorizer, test_texts)
    
    # Extract labels
    y_train = train_df['label'].tolist()
    y_val = val_df['label'].tolist()
    y_test = test_df['label'].tolist()
    
    # Step 6: Configure MLflow
    print("\n[Step 6] Configuring MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Step 7: Train the model and log to MLflow
    print("\n[Step 7] Training model with MLflow tracking...")
    
    # Model hyperparameters
    model_params = {
        'C': 1.0,  # Regularization strength (inverse of lambda)
        'max_iter': 1000,  # Maximum iterations for convergence
        'random_state': 42,
        'solver': 'lbfgs',  # Solver algorithm
        'multi_class': 'multinomial'  # For multi-class classification
    }
    
    with mlflow.start_run():
        # Log hyperparameters
        print("  Logging hyperparameters...")
        mlflow.log_params(model_params)
        mlflow.log_param("vectorizer_max_features", 5000)
        mlflow.log_param("vectorizer_ngram_range", "(1, 2)")
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("val_size", len(val_df))
        mlflow.log_param("test_size", len(test_df))
        
        # Create and train the model
        print("  Training LogisticRegression classifier...")
        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)
        
        # Step 8: Evaluate on validation set
        print("\n[Step 8] Evaluating on validation set...")
        y_val_pred = model.predict(X_val)
        val_metrics = evaluate_classification_model(y_val, y_val_pred, labels=LABELS)
        
        # Log validation metrics
        print("  Validation metrics:")
        for metric_name, metric_value in val_metrics.items():
            if isinstance(metric_value, float):
                mlflow.log_metric(f"val_{metric_name}", metric_value)
                print(f"    {metric_name}: {metric_value:.4f}")
        
        # Print detailed classification report
        print_classification_report(y_val, y_val_pred, labels=LABELS)
        
        # Step 9: Evaluate on test set
        print("\n[Step 9] Evaluating on test set...")
        y_test_pred = model.predict(X_test)
        test_metrics = evaluate_classification_model(y_test, y_test_pred, labels=LABELS)
        
        # Log test metrics
        print("  Test metrics:")
        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, float):
                mlflow.log_metric(f"test_{metric_name}", metric_value)
                print(f"    {metric_name}: {metric_value:.4f}")
        
        # Print detailed classification report
        print_classification_report(y_test, y_test_pred, labels=LABELS)
        
        # Step 10: Log the model and vectorizer to MLflow
        print("\n[Step 10] Logging model and artifacts to MLflow...")
        
        # Log the trained model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME
        )
        
        # Save and log the vectorizer as an artifact
        # We'll save it temporarily and log it
        vectorizer_path = "vectorizer.pkl"
        joblib.dump(vectorizer, vectorizer_path)
        mlflow.log_artifact(vectorizer_path, "vectorizer")
        
        # Clean up temporary file
        if os.path.exists(vectorizer_path):
            os.remove(vectorizer_path)
        
        print("  Model and vectorizer logged successfully!")
        
        # Step 11: Register the model in Model Registry
        print("\n[Step 11] Registering model in MLflow Model Registry...")
        # The model is already registered via log_model with registered_model_name
        # We can optionally promote it to "Production" stage manually or via code
        
        # Get the latest version of the registered model
        try:
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(MODEL_NAME, stages=[])[-1]
            print(f"  Model registered as version {latest_version.version}")
            print(f"  Model URI: {latest_version.source}")
        except Exception as e:
            print(f"  Note: Could not retrieve model version info: {e}")
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        print(f"\nTo view results, start MLflow UI:")
        print(f"  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
        print(f"\nOr access via Docker Compose at http://localhost:5000")


if __name__ == "__main__":
    train_model()

