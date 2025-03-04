#!/usr/bin/env python3
"""
Main entry point for the Prisoner's Dilemma DRL Interactive Demo.
This script launches the Streamlit application.
"""

import os
import sys
import subprocess

def main():
    """Launch the Streamlit application."""
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to the Python path
    sys.path.insert(0, current_dir)
    
    # Launch the Streamlit app
    app_path = os.path.join(current_dir, 'app', 'app.py')
    subprocess.run(['streamlit', 'run', app_path], check=True)

if __name__ == "__main__":
    main() 