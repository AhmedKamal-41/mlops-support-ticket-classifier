"""
Streamlit Cloud entry point.
This file redirects to the main dashboard app.
"""
import sys
from pathlib import Path

# Add dashboard to path and run the main app
sys.path.insert(0, str(Path(__file__).parent / "dashboard"))

# Import and run the main function from dashboard/app.py
from app import main

if __name__ == "__main__":
    main()

