#!/usr/bin/env python3
"""
Main training script for learning piece-square table values from chess game data.
This script demonstrates the complete pipeline from data processing to model training
and integration with the chess engine.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

from wake.ai.learning.data_pipeline import run_data_pipeline, TrainingDataLoader
from wake.ai.learning.pst_models import (
    PSTTrainer, LinearPSTLearner, NeuralPSTLearner, 
    EnsemblePSTLearner, create_model_ensemble
)
from wake.constants import Piece

def main():
    parser = argparse.ArgumentParser(description="Train piece-square tables from chess game data")
    parser.add_argument("--pgn-files", nargs="+", required=True, 
                       help="List of PGN files to process")
    parser.add_argument("--output-dir", default="./pst_training_output",
                       help="Directory for training outputs")
    parser.add_argument("--model-types", nargs="+", 
                       choices=["linear", "neural", "ensemble"], 
                       default=["linear", "neural"],
                       help="Types of models to train")
    parser.add_argument("--skip-processing", action="store_true",
                       help="Skip data processing (use existing training data)")
    
    args = parser.parse_args()
    
    print("🚀 Starting Piece-Square Table Learning Pipeline")
    print("=" * 60)
    
    # Step 1: Data Processing
    if not args.skip_processing:
        print("\n📊 Step 1: Processing game data...")
        data_dir = run_data_pipeline(args.pgn_files, args.output_dir + "/training_data")
    else:
        data_dir = args.output_dir + "/training_data"
        print(f"\n📊 Step 1: Using existing training data from {data_dir}")
    
    # Step 2: Load Training Data
    print("\n📦 Step 2: Loading training data...")
    loader = TrainingDataLoader(data_dir)
    training_data = loader.load_all_data()
    
    print(f"  • Loaded {training_data['samples']} training samples")
    print(f"  • Feature vector size: {training_data['features'].shape[1]}")
    
    # Step 3: Train Models
    print("\n🧠 Step 3: Training models...")
    trainer = PSTTrainer()
    
    # Train different model types
    for model_type in args.model_types:
        print(f"\n  Training {model_type} model...")
        
        if model_type == "linear":
            # Train multiple linear models with different regularization
            models = [
                ("linear_ridge", LinearPSTLearner(regularization="ridge", alpha=1.0)),
                ("linear_lasso", LinearPSTLearner(regularization="lasso", alpha=0.1)),
                ("linear_elastic", LinearPSTLearner(regularization="ridge", alpha=0.5))
            ]
            
        elif model_type == "neural":
            # Train neural networks with different architectures
            models = [
                ("neural_small", NeuralPSTLearner(hidden_layers=[256, 128])),
                ("neural_large", NeuralPSTLearner(hidden_layers=[512, 256, 128]))
            ]
            
        elif model_type == "ensemble":
            # Create ensemble model
            ensemble_models = create_model_ensemble()
            models = [("ensemble", EnsemblePSTLearner(ensemble_models))]
        
        # Train each model variant
        for model_name, model in models:
            print(f"    • Training {model_name}...")
            try:
                metrics = trainer.train_model(model_name, model, training_data)
                print(f"      ✅ R² Score: {metrics.r2:.4f}, MSE: {metrics.mse:.4f}")
            except Exception as e:
                print(f"      ❌ Failed: {str(e)}")
    
    # Step 4: Model Comparison
    print("\n📈 Step 4: Comparing model performance...")
    comparison = trainer.compare_models()
    
    if hasattr(comparison, 'to_string'):  # pandas DataFrame
        print(comparison.to_string())
    else:  # dict fallback
        for name, metrics in comparison.items():
            print(f"  {name}: R²={metrics.get('R²', 'N/A'):.4f}")
    
    # Step 5: Generate Final Tables
    print("\n🏆 Step 5: Generating final piece-square tables...")
    best_model_name, best_model = trainer.get_best_model()
    print(f"  • Best model: {best_model_name}")
    
    final_tables = trainer.generate_final_tables()
    
    # Step 6: Save Results
    print("\n💾 Step 6: Saving results...")
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save models
    models_dir = output_path / "models"
    models_dir.mkdir(exist_ok=True)
    
    for model_name, model in trainer.models.items():
        try:
            model.save_model(str(models_dir / f"{model_name}.pkl"))
            print(f"  • Saved {model_name} model")
        except Exception as e:
            print(f"  • Failed to save {model_name}: {str(e)}")
    
    # Save piece-square tables
    save_piece_square_tables(final_tables, output_path / "learned_pst.py")
    
    # Save training summary
    save_training_summary(trainer, training_data, output_path / "training_summary.json")
    
    print("\n✅ Training pipeline complete!")
    print(f"📁 Results saved to: {output_path}")
    print("\n🔧 Integration:")
    print(f"  • Copy {output_path}/learned_pst.py to wake/ai/constants/")
    print(f"  • Update your engine to use the learned tables")

def save_piece_square_tables(tables: Dict[int, np.ndarray], output_file: Path):
    """Save learned piece-square tables as a Python module"""
    
    piece_names = {
        Piece.wP: "WHITE_PAWN", Piece.bP: "BLACK_PAWN",
        Piece.wN: "WHITE_KNIGHT", Piece.bN: "BLACK_KNIGHT", 
        Piece.wB: "WHITE_BISHOP", Piece.bB: "BLACK_BISHOP",
        Piece.wR: "WHITE_ROOK", Piece.bR: "BLACK_ROOK",
        Piece.wQ: "WHITE_QUEEN", Piece.bQ: "BLACK_QUEEN",
        Piece.wK: "WHITE_KING", Piece.bK: "BLACK_KING"
    }
    
    with open(output_file, 'w') as f:
        f.write('"""\n')
        f.write('Learned piece-square tables from chess game data.\n')
        f.write('Generated automatically by the PST learning pipeline.\n')
        f.write('"""\n\n')
        f.write('import numpy as np\n')
        f.write('from wake.constants import Piece\n\n')
        
        # Write individual tables
        for piece, table in tables.items():
            if piece in piece_names:
                f.write(f"{piece_names[piece]}_TABLE = [\n")
                for row in table:
                    values = ", ".join(f"{val:6.1f}" for val in row)
                    f.write(f"    [{values}],\n")
                f.write("]\n\n")
        
        # Write combined mapping
        f.write("LEARNED_PIECE_TABLES = {\n")
        for piece, name in piece_names.items():
            if piece in tables:
                f.write(f"    Piece.{name.split('_')[1].lower()}{name.split('_')[0][0].lower()}: {name}_TABLE,\n")
        f.write("}\n")
    
    print(f"  • Saved piece-square tables to {output_file}")

def save_training_summary(trainer: PSTTrainer, training_data: Dict, output_file: Path):
    """Save training summary and metrics"""
    
    summary = {
        "training_samples": training_data['samples'],
        "feature_dimensions": training_data['features'].shape[1],
        "models_trained": len(trainer.models),
        "model_performance": {}
    }
    
    # Add model performance metrics
    for name, metrics in trainer.training_history.items():
        summary["model_performance"][name] = {
            "r2_score": float(metrics.r2),
            "mse": float(metrics.mse),
            "mae": float(metrics.mae),
            "cross_val_score": float(metrics.cross_val_score),
            "training_samples": int(metrics.training_samples)
        }
    
    # Best model info
    best_name, _ = trainer.get_best_model()
    summary["best_model"] = best_name
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  • Saved training summary to {output_file}")

def create_integration_example():
    """Create an example of how to integrate learned tables into the engine"""
    
    example_code = '''
# Example: Integrating learned piece-square tables into your engine

from wake.ai.constants.learned_pst import LEARNED_PIECE_TABLES
from wake.constants import Piece

class ImprovedEvaluator:
    """Chess position evaluator using learned piece-square tables"""
    
    def __init__(self, use_learned_tables=True):
        self.piece_tables = LEARNED_PIECE_TABLES if use_learned_tables else DEFAULT_TABLES
    
    def evaluate_position(self, position):
        """Evaluate a chess position using learned heuristics"""
        score = 0
        
        for piece_type, squares in position.piece_map.items():
            table = self.piece_tables.get(piece_type)
            if table is not None:
                for square in squares:
                    row, col = square // 8, square % 8
                    # For black pieces, flip the table vertically
                    if piece_type in Piece.black_pieces:
                        row = 7 - row
                    score += table[row][col]
        
        return score
    
    def get_piece_value_on_square(self, piece_type, square):
        """Get the learned value for a piece on a specific square"""
        table = self.piece_tables.get(piece_type)
        if table is None:
            return 0
        
        row, col = square // 8, square % 8
        if piece_type in Piece.black_pieces:
            row = 7 - row
        
        return table[row][col]

# Usage in your chess engine:
evaluator = ImprovedEvaluator(use_learned_tables=True)
position_score = evaluator.evaluate_position(current_position)
'''
    
    return example_code

if __name__ == "__main__":
    main() 