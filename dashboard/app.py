"""
Premium MLOps Support Ticket Classifier Dashboard

A professional, recruiter-ready monitoring dashboard with 5 tabs:
- Overview: KPIs and key metrics
- Model Quality: Performance metrics and confusion matrix
- Monitoring: Inference analytics with filters
- Drift: Data drift detection reports
- Runbook: Operational guides
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime, timedelta
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PROJECT_ROOT, LABELS

# Import UI components
from dashboard.ui_components import (
    render_kpi_card,
    render_status_badge,
    render_section_header,
    render_empty_state,
    render_metric_with_color
)

# Page configuration
st.set_page_config(
    page_title="MLOps Ticket Classifier",
    page_icon="📊",
    layout="wide"
)

# Load CSS
def load_css():
    """Load custom CSS styles."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Initialize session state
if 'paths' not in st.session_state:
    st.session_state.paths = {
        'classification_report': None,
        'confusion_matrix': None,
        'inference_logs': None,
        'drift_report': None
    }


# Data loading functions with caching
@st.cache_data
def load_classification_report(report_path: Path) -> dict:
    """Load classification report from JSON file."""
    if not report_path or not report_path.exists():
        return None
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        return None


@st.cache_data
def load_inference_logs(logs_file: Path) -> pd.DataFrame:
    """Load inference logs from JSONL file."""
    if not logs_file or not logs_file.exists():
        return None
    
    logs = []
    try:
        with open(logs_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        log_entry = json.loads(line)
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        
        if not logs:
            return None
        
        df = pd.DataFrame(logs)
        
        # Parse timestamp if it's a string
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception:
        return None


@st.cache_data
def load_confusion_matrix_image(cm_path: Path):
    """Load confusion matrix PNG image."""
    if not cm_path or not cm_path.exists():
        return None
    try:
        return Image.open(str(cm_path))
    except Exception:
        return None


@st.cache_data
def load_drift_report(drift_path: Path) -> dict:
    """Load drift detection report from JSON file."""
    if not drift_path or not drift_path.exists():
        return None
    try:
        with open(drift_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def get_status_and_last_updated(paths: dict) -> tuple:
    """Compute system status and last updated timestamp."""
    artifacts = [
        paths.get('classification_report'),
        paths.get('confusion_matrix'),
        paths.get('inference_logs'),
    ]
    
    existing = [p for p in artifacts if p and Path(p).exists()]
    
    if len(existing) == 0:
        status = "Missing Artifacts"
        last_updated = None
    elif len(existing) == len(artifacts):
        status = "Healthy"
        last_updated = max(Path(p).stat().st_mtime for p in existing)
    else:
        status = "Degraded"
        last_updated = max(Path(p).stat().st_mtime for p in existing) if existing else None
    
    if last_updated:
        last_updated_str = datetime.fromtimestamp(last_updated).strftime("%Y-%m-%d %H:%M")
    else:
        last_updated_str = "Never"
    
    return status, last_updated_str


def render_header_row(status: str, last_updated: str):
    """Render dashboard header with title and status."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="dashboard-header">
            <div>
                <div class="dashboard-title">MLOps Ticket Classifier</div>
                <div class="dashboard-subtitle">Real-time Monitoring & Analytics Dashboard</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="status-info">Last Updated: {last_updated}</div>', unsafe_allow_html=True)
        render_status_badge(status)


def sidebar_config():
    """Render sidebar configuration."""
    st.sidebar.header("Configuration")
    
    st.sidebar.subheader("File Paths")
    
    # Default paths
    default_reports_dir = PROJECT_ROOT / "reports"
    default_logs_dir = PROJECT_ROOT / "logs"
    
    # Path inputs
    report_path = st.sidebar.text_input(
        "Classification Report",
        value=str(default_reports_dir / "classification_report.json")
    )
    
    cm_path = st.sidebar.text_input(
        "Confusion Matrix",
        value=str(default_reports_dir / "confusion_matrix.png")
    )
    
    logs_path = st.sidebar.text_input(
        "Inference Logs",
        value=str(default_logs_dir / "inference_logs.jsonl")
    )
    
    drift_path = st.sidebar.text_input(
        "Drift Report (Optional)",
        value=str(default_reports_dir / "drift_report.json")
    )
    
    # Update session state
    st.session_state.paths = {
        'classification_report': Path(report_path) if report_path else None,
        'confusion_matrix': Path(cm_path) if cm_path else None,
        'inference_logs': Path(logs_path) if logs_path else None,
        'drift_report': Path(drift_path) if drift_path else None
    }


def render_overview_tab(report: dict, df_logs: pd.DataFrame):
    """Render Overview tab."""
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    # Requests (24h)
    requests_24h = 0
    if df_logs is not None and not df_logs.empty:
        if 'timestamp' in df_logs.columns:
            cutoff = datetime.now() - timedelta(hours=24)
            recent = df_logs[df_logs['timestamp'] >= cutoff]
            requests_24h = len(recent)
    
    with col1:
        render_kpi_card("Requests (24h)", f"{requests_24h:,}")
    
    # p95 Latency
    p95_latency = "N/A"
    if df_logs is not None and not df_logs.empty and 'latency_ms' in df_logs.columns:
        latency_data = df_logs['latency_ms'].dropna()
        if not latency_data.empty:
            p95_latency = f"{latency_data.quantile(0.95):.1f}"
    
    with col2:
        render_kpi_card("p95 Latency (ms)", p95_latency)
    
    # Error Rate
    error_rate = "0.0"
    if df_logs is not None and not df_logs.empty and 'status_code' in df_logs.columns:
        total = len(df_logs)
        errors = len(df_logs[df_logs['status_code'] == 500])
        error_rate = f"{(errors / total * 100):.1f}"
    
    with col3:
        render_kpi_card("Error Rate (%)", error_rate)
    
    # Macro-F1
    macro_f1 = "N/A"
    if report and 'summary' in report:
        macro_f1 = f"{report['summary'].get('macro_f1', 0):.4f}"
    
    with col4:
        render_kpi_card("Macro-F1", macro_f1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two-column row
    col_left, col_right = st.columns(2)
    
    with col_left:
        render_section_header("Request & Latency Trends")
        
        if df_logs is not None and not df_logs.empty and 'timestamp' in df_logs.columns:
            # Requests over time
            df_logs_copy = df_logs.copy()
            df_logs_copy['minute'] = df_logs_copy['timestamp'].dt.floor('min')
            rpm_df = df_logs_copy.groupby('minute').size().reset_index(name='requests')
            rpm_df = rpm_df.sort_values('minute')
            
            fig = px.line(
                rpm_df,
                x='minute',
                y='requests',
                title='Requests per Minute',
                labels={'minute': 'Time', 'requests': 'Requests'}
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Latency p50/p95 over time
            if 'latency_ms' in df_logs_copy.columns:
                latency_df = df_logs_copy.groupby('minute')['latency_ms'].agg(['median', lambda x: x.quantile(0.95)]).reset_index()
                latency_df.columns = ['minute', 'p50', 'p95']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=latency_df['minute'],
                    y=latency_df['p50'],
                    name='p50',
                    line=dict(color='#3b82f6')
                ))
                fig.add_trace(go.Scatter(
                    x=latency_df['minute'],
                    y=latency_df['p95'],
                    name='p95',
                    line=dict(color='#ef4444')
                ))
                fig.update_layout(
                    title='Latency Percentiles Over Time',
                    xaxis_title='Time',
                    yaxis_title='Latency (ms)',
                    height=300,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            render_empty_state("No inference logs available", "Generate Sample Logs")
    
    with col_right:
        render_section_header("Predictions & Recent Activity")
        
        if df_logs is not None and not df_logs.empty:
            # Prediction distribution
            if 'predicted_label' in df_logs.columns:
                pred_counts = df_logs['predicted_label'].value_counts().reset_index()
                pred_counts.columns = ['label', 'count']
                
                fig = px.bar(
                    pred_counts,
                    x='label',
                    y='count',
                    title='Prediction Distribution',
                    labels={'label': 'Label', 'count': 'Count'}
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent requests table
            st.markdown("**Recent Requests (Last 50)**")
            recent_df = df_logs.tail(50)[['timestamp', 'predicted_label', 'latency_ms', 'status_code']].copy()
            if 'timestamp' in recent_df.columns:
                recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
        else:
            render_empty_state("No inference logs available", "Generate Sample Logs")


def render_model_quality_tab(report: dict, cm_image, df_logs: pd.DataFrame):
    """Render Model Quality tab."""
    
    if not report:
        render_empty_state(
            "No classification report found",
            "Generate Sample Reports",
            lambda: None
        )
        return
    
    # Metrics Table
    render_section_header("Per-Class Performance Metrics")
    
    if 'per_class' in report:
        class_data = []
        for label, metrics in report['per_class'].items():
            f1 = metrics.get('f1', 0.0)
            class_data.append({
                'Label': label.title(),
                'Precision': f"{metrics.get('precision', 0.0):.4f}",
                'Recall': f"{metrics.get('recall', 0.0):.4f}",
                'F1 Score': f"{f1:.4f}",
                'Support': metrics.get('support', 0)
            })
        
        df_metrics = pd.DataFrame(class_data)
        
        # Color code F1 scores in display
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Confusion Matrix
    render_section_header("Confusion Matrix")
    
    if cm_image:
        st.image(cm_image, width=600, caption="Confusion Matrix")
    else:
        render_empty_state("Confusion matrix not available")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Top Misclassified Examples
    render_section_header("Top Misclassified Examples")
    
    if df_logs is not None and not df_logs.empty:
        # Check if we have ground truth labels
        required_cols = ['text', 'predicted_label', 'true_label']
        if all(col in df_logs.columns for col in required_cols):
            misclassified = df_logs[df_logs['predicted_label'] != df_logs['true_label']].copy()
            
            if not misclassified.empty:
                # Add confidence if available
                if 'confidence' in misclassified.columns:
                    misclassified = misclassified.sort_values('confidence', ascending=True).head(20)
                else:
                    misclassified = misclassified.head(20)
                
                # Prepare display
                display_df = misclassified[['text', 'true_label', 'predicted_label']].copy()
                display_df['text'] = display_df['text'].str[:100] + '...'  # Truncate
                display_df.columns = ['Text (truncated)', 'True Label', 'Predicted Label']
                
                if 'confidence' in misclassified.columns:
                    display_df['Confidence'] = misclassified['confidence'].apply(lambda x: f"{x:.3f}")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No misclassified examples found in logs.")
        else:
            st.info("Misclassified examples require 'text', 'predicted_label', and 'true_label' columns in logs.")
    else:
        render_empty_state(
            "No inference logs available to analyze misclassifications",
            "Generate Sample Logs",
            generate_sample_logs_callback
        )


def render_monitoring_tab(df_logs: pd.DataFrame):
    """Render Monitoring tab with filters and analytics."""
    
    # Sidebar filters
    st.sidebar.subheader("Filters")
    
    time_ranges = {
        "Last 1h": timedelta(hours=1),
        "Last 6h": timedelta(hours=6),
        "Last 24h": timedelta(hours=24),
        "Last 7d": timedelta(days=7),
        "Last 30d": timedelta(days=30),
        "All": None
    }
    
    selected_range = st.sidebar.selectbox("Time Range", list(time_ranges.keys()))
    
    # Class filter
    selected_classes = st.sidebar.multiselect(
        "Class Filter",
        options=LABELS,
        default=LABELS
    )
    
    # Status code filter
    status_options = ["All", "200", "500"]
    selected_status = st.sidebar.selectbox("Status Code", status_options)
    
    if df_logs is None or df_logs.empty:
        render_empty_state("No inference logs found", "Generate Sample Logs")
        return
    
    # Apply filters
    filtered_df = df_logs.copy()
    
    # Time filter
    if time_ranges[selected_range] and 'timestamp' in filtered_df.columns:
        cutoff = pd.Timestamp.now() - time_ranges[selected_range]
        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(filtered_df['timestamp']):
            filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
        filtered_df = filtered_df[filtered_df['timestamp'] >= cutoff]
    
    # Class filter
    if selected_classes and 'predicted_label' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['predicted_label'].isin(selected_classes)]
    
    # Status filter
    if selected_status != "All" and 'status_code' in filtered_df.columns:
        # Handle both string and int status codes
        status_val = int(selected_status)
        filtered_df = filtered_df[filtered_df['status_code'] == status_val]
    
    # Show filter info
    if filtered_df.empty:
        st.warning(f"No data matches the selected filters. Original dataset has {len(df_logs)} rows.")
        return
    
    st.info(f"Showing {len(filtered_df)} of {len(df_logs)} requests (filtered)")
    
    # Throughput Chart
    render_section_header("Throughput")
    
    if 'timestamp' in filtered_df.columns and not filtered_df.empty:
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['minute'] = filtered_df_copy['timestamp'].dt.floor('min')
        rpm_df = filtered_df_copy.groupby('minute').size().reset_index(name='requests')
        rpm_df = rpm_df.sort_values('minute')
        
        fig = px.line(
            rpm_df,
            x='minute',
            y='requests',
            title='Requests per Minute',
            labels={'minute': 'Time', 'requests': 'Requests'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Latency Histogram
    render_section_header("Latency Distribution")
    
    if 'latency_ms' in filtered_df.columns and not filtered_df.empty:
        latency_data = filtered_df['latency_ms'].dropna()
        if not latency_data.empty:
            # Convert to DataFrame for plotly
            latency_df = pd.DataFrame({'latency_ms': latency_data})
            fig = px.histogram(
                latency_df,
                x='latency_ms',
                nbins=50,
                title='Latency Distribution',
                labels={'latency_ms': 'Latency (ms)', 'count': 'Frequency'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No latency data available after filtering.")
    else:
        st.info("Latency data not available.")
    
    # Errors Over Time
    render_section_header("Errors Over Time")
    
    if 'status_code' in filtered_df.columns and 'timestamp' in filtered_df.columns and not filtered_df.empty:
        errors_df = filtered_df[filtered_df['status_code'] == 500].copy()
        if not errors_df.empty:
            errors_df['hour'] = errors_df['timestamp'].dt.floor('h')
            error_counts = errors_df.groupby('hour').size().reset_index(name='errors')
            
            fig = px.line(
                error_counts,
                x='hour',
                y='errors',
                title='500 Errors Over Time',
                labels={'hour': 'Time', 'errors': 'Error Count'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No errors found in the selected time range.")
    
    # Searchable Inference Logs Table
    render_section_header("Inference Logs")
    
    if filtered_df.empty:
        st.info("No data to display after applying filters.")
        return
    
    display_cols = ['timestamp', 'predicted_label', 'latency_ms', 'status_code']
    if 'text' in filtered_df.columns:
        display_cols.insert(1, 'text')
    
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    
    if not available_cols:
        st.warning("No displayable columns found in filtered data.")
        return
    
    display_df = filtered_df[available_cols].copy()
    
    if 'timestamp' in display_df.columns:
        # Ensure timestamp is datetime before formatting
        if pd.api.types.is_datetime64_any_dtype(display_df['timestamp']):
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    if 'text' in display_df.columns:
        display_df['text'] = display_df['text'].astype(str).str[:50] + '...'  # Truncate for display
    
    st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)


def render_drift_tab(drift_report: dict):
    """Render Drift tab."""
    
    if drift_report:
        render_section_header("Drift Detection Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            drift_score = drift_report.get('drift_score', 0)
            status = drift_report.get('status', 'unknown')
            
            render_kpi_card("Drift Score", f"{drift_score:.3f}")
            st.markdown(f"**Status:** {status.title()}")
        
        with col2:
            if 'reference_stats' in drift_report and 'recent_stats' in drift_report:
                ref_stats = drift_report['reference_stats']
                recent_stats = drift_report['recent_stats']
                
                st.markdown("**Reference Statistics**")
                st.json(ref_stats)
                
                st.markdown("**Recent Statistics**")
                st.json(recent_stats)
        
        # Visual comparison
        if 'reference_stats' in drift_report and 'recent_stats' in drift_report:
            ref_mean = drift_report['reference_stats'].get('mean', 0)
            recent_mean = drift_report['recent_stats'].get('mean', 0)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Reference', 'Recent'],
                y=[ref_mean, recent_mean],
                marker_color=['#3b82f6', '#ef4444']
            ))
            fig.update_layout(
                title='Mean Comparison: Reference vs Recent',
                yaxis_title='Mean Value',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        render_empty_state(
            "No drift report found",
            "Generate Sample Drift Report",
            lambda: None
        )
        
        if st.button("Generate Sample Drift Report", type="primary"):
            import subprocess
            script_path = PROJECT_ROOT / "scripts" / "generate_sample_reports.py"
            try:
                subprocess.run(
                    [sys.executable, str(script_path), "--include-drift"],
                    cwd=str(PROJECT_ROOT),
                    timeout=30
                )
                st.success("Drift report generated! Refresh to see it.")
                st.rerun()
            except Exception as e:
                st.error(f"Error generating drift report: {e}")


def render_runbook_tab():
    """Render Runbook tab with operational guides."""
    
    render_section_header("Operational Runbook")
    
    with st.expander("🚀 Training", expanded=False):
        st.markdown("""
        **Train a new model:**
        ```bash
        python -m src.train
        ```
        
        This will:
        - Validate the dataset
        - Train a LogisticRegression classifier
        - Evaluate on validation and test sets
        - Save reports to `reports/` directory
        - Log everything to MLflow
        """)
    
    with st.expander("📊 Evaluation", expanded=False):
        st.markdown("""
        **Evaluate a trained model:**
        ```bash
        python -m src.evaluate --split
        ```
        
        Options:
        - `--split`: Evaluate on test split (default: full dataset)
        - `--output`: Custom output directory
        - `--mlflow-run-id`: Evaluate specific MLflow run
        """)
    
    with st.expander("🌐 Serving", expanded=False):
        st.markdown("""
        **Start the FastAPI inference server:**
        ```bash
        uvicorn src.api.main:app --reload
        ```
        
        API endpoints:
        - `POST /predict` - Make predictions
        - `GET /health` - Health check
        - `GET /metrics` - Prometheus metrics
        - `GET /drift/stats` - Drift detection stats
        """)
    
    with st.expander("📈 MLflow", expanded=False):
        st.markdown("""
        **Start MLflow UI:**
        ```bash
        mlflow ui --backend-store-uri file:///./mlruns
        ```
        
        Access at: http://localhost:5000
        
        View:
        - Training runs and metrics
        - Model versions
        - Evaluation artifacts
        - Compare experiments
        """)
    
    with st.expander("📉 Drift Detection", expanded=False):
        st.markdown("""
        **Check drift via API:**
        ```bash
        curl http://localhost:8000/drift/stats
        ```
        
        **Generate drift report:**
        The drift detector runs automatically when serving predictions.
        Generate a sample drift report:
        ```bash
        python scripts/generate_sample_reports.py --include-drift
        ```
        """)
    
    with st.expander("📊 Dashboard", expanded=False):
        st.markdown("""
        **Start the dashboard:**
        ```bash
        streamlit run dashboard/app.py
        ```
        
        **Generate sample data:**
        ```bash
        # Generate sample logs
        python scripts/generate_sample_logs.py --num-logs 1000
        
        # Generate sample reports
        python scripts/generate_sample_reports.py --include-drift
        ```
        
        Access at: http://localhost:8501
        """)


def generate_sample_logs_callback():
    """Callback to generate sample logs."""
    import subprocess
    script_path = PROJECT_ROOT / "scripts" / "generate_sample_logs.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            st.success("Sample logs generated successfully!")
            st.cache_data.clear()
        else:
            st.error(f"Error: {result.stderr}")
    except Exception as e:
        st.error(f"Error generating logs: {e}")


def main():
    """Main dashboard application."""
    
    # Sidebar configuration
    sidebar_config()
    
    # Get paths from session state
    paths = st.session_state.paths
    
    # Load data
    report = load_classification_report(paths['classification_report'])
    df_logs = load_inference_logs(paths['inference_logs'])
    cm_image = load_confusion_matrix_image(paths['confusion_matrix'])
    drift_report = load_drift_report(paths['drift_report'])
    
    # Compute status
    status, last_updated = get_status_and_last_updated(paths)
    
    # Render header
    render_header_row(status, last_updated)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Model Quality",
        "Monitoring",
        "Drift",
        "Runbook"
    ])
    
    with tab1:
        render_overview_tab(report, df_logs)
    
    with tab2:
        render_model_quality_tab(report, cm_image, df_logs)
    
    with tab3:
        render_monitoring_tab(df_logs)
    
    with tab4:
        render_drift_tab(drift_report)
    
    with tab5:
        render_runbook_tab()


if __name__ == "__main__":
    main()
