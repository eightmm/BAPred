#!/usr/bin/env python3
"""
Command-line interface for BAPred
"""
import sys
import os

def main():
    """Main CLI entry point"""
    # Import here to avoid circular imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_inference import main as run_main
    run_main()

if __name__ == "__main__":
    main()