"""
Streamlit Cloud entry point.
This file redirects to the main dashboard app.
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root and dashboard to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "dashboard"))

# Import and run the main function from dashboard/app.py
try:
    from dashboard.app import main
    main()
except ImportError:
    # Fallback: try direct import
    import importlib.util
    app_path = project_root / "dashboard" / "app.py"
    spec = importlib.util.spec_from_file_location("dashboard.app", app_path)
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    app_module.main()

