#!/usr/bin/env python3
"""
BA-Pred Inference Script
A convenient wrapper for running binding affinity prediction.
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from bapred.inference import inference
from bapred.logger import setup_logger
import argparse
import torch

def main():
    # Setup logger
    logger = setup_logger("bapred.cli")
    parser = argparse.ArgumentParser(
        description='BA-Pred: Protein-ligand Binding Affinity Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py -r example/1KLT.pdb -l example/ligands.sdf -o result.csv
  python run_inference.py -r example/1KLT.pdb -l example/ligands.sdf -o result.csv --device cpu
  python run_inference.py -r example/1KLT.pdb -l example/ligands.sdf -o result.csv --batch_size 64
        """
    )
    
    parser.add_argument('-r', '--protein_pdb', required=True, 
                       help='Receptor protein PDB file path')
    parser.add_argument('-l', '--ligand_file', required=True,
                       help='Ligand file path (.sdf/.mol2/.txt)')
    parser.add_argument('-o', '--output', required=True,
                       help='Output CSV file path for results')
    
    parser.add_argument('--batch_size', default=16, type=int,
                       help='Batch size for inference (default: 128)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'],
                       help='Device to use: cpu or cuda (default: cuda)')
    parser.add_argument('--model_path', type=str, default='./bapred/weight',
                       help='Path to model weights directory (default: ./bapred/weight)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.protein_pdb):
        logger.error(f"Protein PDB file not found: {args.protein_pdb}")
        sys.exit(1)
    
    if not os.path.exists(args.ligand_file):
        logger.error(f"Ligand file not found: {args.ligand_file}")
        sys.exit(1)
    
    if not os.path.exists(f"{args.model_path}/BAPred.pth"):
        logger.error(f"Model weights not found: {args.model_path}/BAPred.pth")
        sys.exit(1)
    
    # Setup device
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
    else:
        device = 'cpu'
        logger.info("Using CPU")
    
    logger.info(f"Input protein: {args.protein_pdb}")
    logger.info(f"Input ligands: {args.ligand_file}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {device}")
    logger.info("-" * 50)
    
    try:
        inference(
            protein_pdb=args.protein_pdb,
            ligand_file=args.ligand_file,
            output=args.output,
            batch_size=args.batch_size,
            model_path=args.model_path,
            device=device
        )
        logger.info(f"\nInference completed successfully!")
        logger.info(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
