"""
Unit tests for BAPred inference module
"""
import unittest
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestInference(unittest.TestCase):
    """Test cases for inference functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path(__file__).parent.parent
        self.example_dir = self.test_dir / "example"
        self.protein_pdb = str(self.example_dir / "1KLT.pdb")
        self.ligand_file = str(self.example_dir / "ligands.sdf")
        
    def test_example_files_exist(self):
        """Test that example files exist"""
        self.assertTrue(os.path.exists(self.protein_pdb), 
                       f"Protein file not found: {self.protein_pdb}")
        self.assertTrue(os.path.exists(self.ligand_file), 
                       f"Ligand file not found: {self.ligand_file}")
    
    def test_import_inference(self):
        """Test that inference module can be imported"""
        try:
            from bapred.inference import inference
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import inference: {e}")
    
    def test_model_weights_exist(self):
        """Test that model weights exist"""
        weight_path = self.test_dir / "bapred" / "weight" / "BAPred.pth"
        self.assertTrue(weight_path.exists(), 
                       f"Model weights not found: {weight_path}")
    
    def test_cli_help(self):
        """Test CLI help command"""
        import subprocess
        result = subprocess.run(
            [sys.executable, "run_inference.py", "--help"],
            cwd=self.test_dir,
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0, 
                        f"CLI help failed: {result.stderr}")
        self.assertIn("BAPred", result.stdout)
        self.assertIn("--ncpu", result.stdout)

if __name__ == "__main__":
    unittest.main()