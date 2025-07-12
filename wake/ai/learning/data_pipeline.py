"""
Data processing pipeline for converting chess game databases into training data
for piece-square table learning. Handles PGN files, game filtering, and feature extraction.
"""

import os
import re
import gzip
from typing import Iterator, Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from pathlib import Path

try:
    import chess
    import chess.pgn
    PYTHON_CHESS_AVAILABLE = True
except ImportError:
    PYTHON_CHESS_AVAILABLE = False

from wake.position import Position
from wake.move import Move
from wake.constants import Piece, Color
from wake.ai.learning.feature_extractor import FeatureExtractor, GameContext

@dataclass
class GameMetadata:
    """Metadata extracted from a chess game"""
    white_elo: int
    black_elo: int
    result: str  # "1-0", "0-1", "1/2-1/2"
    time_control: str
    opening: str
    moves_count: int
    game_id: str

@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline"""
    min_elo: int = 1500
    max_elo: int = 3000
    min_moves: int = 20
    max_moves: int = 200
    sample_interval: int = 3  # Extract position every N moves
    max_games_per_file: int = 10000
    parallel_workers: int = 4
    filter_time_controls: List[str] = None  # e.g., ["blitz", "rapid", "classical"]
    exclude_results: List[str] = None  # e.g., ["*"] to exclude unfinished games

class PGNProcessor:
    """Processes PGN files and extracts training data"""
    
    def __init__(self, config: ProcessingConfig = None):
        if not PYTHON_CHESS_AVAILABLE:
            raise ImportError("python-chess required for PGN processing")
        
        self.config = config or ProcessingConfig()
        self.feature_extractor = FeatureExtractor()
        
    def process_pgn_file(self, pgn_path: str, output_dir: str) -> Dict[str, Any]:
        """Process a single PGN file and save training data"""
        
        print(f"Processing {pgn_path}...")
        
        # Handle compressed files
        if pgn_path.endswith('.gz'):
            pgn_file = gzip.open(pgn_path, 'rt')
        else:
            pgn_file = open(pgn_path, 'r')
        
        training_samples = []
        games_processed = 0
        games_skipped = 0
        
        try:
            while games_processed < self.config.max_games_per_file:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                # Filter game based on criteria
                if not self._should_process_game(game):
                    games_skipped += 1
                    continue
                
                # Extract training samples from this game
                game_samples = self._extract_game_samples(game)
                training_samples.extend(game_samples)
                games_processed += 1
                
                if games_processed % 1000 == 0:
                    print(f"  Processed {games_processed} games, "
                          f"extracted {len(training_samples)} samples")
        
        finally:
            pgn_file.close()
        
        # Save training data
        output_file = Path(output_dir) / f"{Path(pgn_path).stem}_training_data.pkl"
        self._save_training_data(training_samples, output_file)
        
        return {
            'file': pgn_path,
            'games_processed': games_processed,
            'games_skipped': games_skipped,
            'samples_extracted': len(training_samples),
            'output_file': str(output_file)
        }
    
    def process_multiple_files(self, pgn_files: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """Process multiple PGN files in parallel"""
        
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            future_to_file = {
                executor.submit(self.process_pgn_file, pgn_file, output_dir): pgn_file
                for pgn_file in pgn_files
            }
            
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)
                print(f"Completed {result['file']}: {result['samples_extracted']} samples")
        
        return results
    
    def _should_process_game(self, game: chess.pgn.Game) -> bool:
        """Check if a game meets processing criteria"""
        headers = game.headers
        
        # Check ELO ratings
        try:
            white_elo = int(headers.get('WhiteElo', 0))
            black_elo = int(headers.get('BlackElo', 0))
            
            if (white_elo < self.config.min_elo or white_elo > self.config.max_elo or
                black_elo < self.config.min_elo or black_elo > self.config.max_elo):
                return False
        except (ValueError, TypeError):
            return False
        
        # Check game result
        result = headers.get('Result', '*')
        if self.config.exclude_results and result in self.config.exclude_results:
            return False
        
        # Check time control
        time_control = headers.get('TimeControl', '').lower()
        if self.config.filter_time_controls:
            if not any(tc in time_control for tc in self.config.filter_time_controls):
                return False
        
        # Check move count (approximate from mainline)
        moves = list(game.mainline_moves())
        if len(moves) < self.config.min_moves or len(moves) > self.config.max_moves:
            return False
        
        return True
    
    def _extract_game_samples(self, game: chess.pgn.Game) -> List[Dict[str, Any]]:
        """Extract training samples from a single game"""
        
        # Get game metadata
        metadata = self._extract_game_metadata(game)
        
        # Convert result to numeric value
        result_map = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
        game_result = result_map.get(metadata.result, 0.5)
        
        samples = []
        board = chess.Board()
        move_number = 0
        
        # Play through the game and extract positions
        for move in game.mainline_moves():
            move_number += 1
            
            # Sample positions at regular intervals
            if move_number % self.config.sample_interval == 0:
                # Convert python-chess board to our Position format
                position = self._chess_board_to_position(board)
                
                # Create game context
                game_context = GameContext(
                    move_number=move_number,
                    game_phase=self._determine_game_phase(move_number, len(list(game.mainline_moves()))),
                    material_balance=self._calculate_material_balance(board),
                    game_result=game_result,
                    player_ratings=(metadata.white_elo, metadata.black_elo),
                    time_control=metadata.time_control
                )
                
                # Extract features and create training sample
                training_sample = self.feature_extractor.create_training_sample(position, game_context)
                samples.append(training_sample)
            
            # Make the move
            board.push(move)
        
        return samples
    
    def _extract_game_metadata(self, game: chess.pgn.Game) -> GameMetadata:
        """Extract metadata from game headers"""
        headers = game.headers
        
        return GameMetadata(
            white_elo=int(headers.get('WhiteElo', 1500)),
            black_elo=int(headers.get('BlackElo', 1500)),
            result=headers.get('Result', '1/2-1/2'),
            time_control=headers.get('TimeControl', 'unknown'),
            opening=headers.get('Opening', 'unknown'),
            moves_count=len(list(game.mainline_moves())),
            game_id=headers.get('Site', '') + '_' + headers.get('Round', '')
        )
    
    def _chess_board_to_position(self, board: chess.Board) -> Position:
        """Convert python-chess Board to our Position class"""
        # This is a simplified conversion - in practice, you'd need to
        # properly map between the different piece representations
        
        position = Position()
        
        # Clear the position
        for piece_type in position.piece_map:
            position.piece_map[piece_type] = set()
        
        # Map pieces from python-chess board
        piece_map = {
            chess.PAWN: {chess.WHITE: Piece.wP, chess.BLACK: Piece.bP},
            chess.KNIGHT: {chess.WHITE: Piece.wN, chess.BLACK: Piece.bN},
            chess.BISHOP: {chess.WHITE: Piece.wB, chess.BLACK: Piece.bB},
            chess.ROOK: {chess.WHITE: Piece.wR, chess.BLACK: Piece.bR},
            chess.QUEEN: {chess.WHITE: Piece.wQ, chess.BLACK: Piece.bQ},
            chess.KING: {chess.WHITE: Piece.wK, chess.BLACK: Piece.bK},
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                our_piece = piece_map[piece.piece_type][piece.color]
                position.piece_map[our_piece].add(square)
        
        # Update mailbox and other state
        position.sync_mailbox_from_piece_map()
        position.color_to_move = Color.WHITE if board.turn == chess.WHITE else Color.BLACK
        
        return position
    
    def _determine_game_phase(self, move_number: int, total_moves: int) -> str:
        """Determine game phase based on move number"""
        if move_number <= 15:
            return 'opening'
        elif move_number <= total_moves * 0.7:
            return 'middlegame'
        else:
            return 'endgame'
    
    def _calculate_material_balance(self, board: chess.Board) -> float:
        """Calculate material balance from current position"""
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return (white_material - black_material) / max(1, white_material + black_material)
    
    def _save_training_data(self, samples: List[Dict[str, Any]], output_file: Path):
        """Save training samples to disk"""
        with open(output_file, 'wb') as f:
            pickle.dump(samples, f)

class TrainingDataLoader:
    """Loads and combines training data from multiple sources"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load_all_data(self) -> Dict[str, np.ndarray]:
        """Load and combine all training data files"""
        
        all_samples = []
        
        # Find all pickle files in data directory
        for pickle_file in self.data_dir.glob("*_training_data.pkl"):
            print(f"Loading {pickle_file}...")
            with open(pickle_file, 'rb') as f:
                samples = pickle.load(f)
                all_samples.extend(samples)
        
        print(f"Loaded {len(all_samples)} total training samples")
        
        # Convert to numpy arrays
        features = np.array([sample['features'] for sample in all_samples])
        outcomes = np.array([sample['outcome'] for sample in all_samples])
        weights = np.array([sample['weight'] for sample in all_samples])
        
        return {
            'features': features,
            'outcomes': outcomes,
            'weights': weights,
            'samples': len(all_samples)
        }
    
    def load_filtered_data(self, 
                          min_rating: int = 1500,
                          game_phases: List[str] = None) -> Dict[str, np.ndarray]:
        """Load training data with additional filtering"""
        
        all_samples = []
        
        for pickle_file in self.data_dir.glob("*_training_data.pkl"):
            with open(pickle_file, 'rb') as f:
                samples = pickle.load(f)
                
                # Apply filters
                filtered_samples = []
                for sample in samples:
                    # Add filtering logic here if needed
                    # For now, just use all samples
                    filtered_samples.append(sample)
                
                all_samples.extend(filtered_samples)
        
        # Convert to numpy arrays
        features = np.array([sample['features'] for sample in all_samples])
        outcomes = np.array([sample['outcome'] for sample in all_samples])
        weights = np.array([sample['weight'] for sample in all_samples])
        
        return {
            'features': features,
            'outcomes': outcomes,
            'weights': weights,
            'samples': len(all_samples)
        }

# Example usage and pipeline runner
def run_data_pipeline(pgn_files: List[str], output_dir: str) -> str:
    """Complete data processing pipeline"""
    
    # Configure processing
    config = ProcessingConfig(
        min_elo=1600,
        max_elo=2800,
        min_moves=25,
        max_moves=150,
        sample_interval=4,
        max_games_per_file=5000,
        parallel_workers=4
    )
    
    # Process PGN files
    processor = PGNProcessor(config)
    results = processor.process_multiple_files(pgn_files, output_dir)
    
    # Print summary
    total_games = sum(r['games_processed'] for r in results)
    total_samples = sum(r['samples_extracted'] for r in results)
    
    print(f"\n=== Data Processing Complete ===")
    print(f"Total games processed: {total_games}")
    print(f"Total training samples: {total_samples}")
    print(f"Output directory: {output_dir}")
    
    return output_dir

# Example for downloading and processing Lichess data
def download_lichess_data(year_month: str = "2023-12") -> str:
    """
    Example function for downloading Lichess open database.
    In practice, you'd use wget or requests to download:
    https://database.lichess.org/standard/lichess_db_standard_rated_{year_month}.pgn.bz2
    """
    
    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{year_month}.pgn.bz2"
    
    print(f"To download Lichess data, run:")
    print(f"wget {url}")
    print(f"bunzip2 lichess_db_standard_rated_{year_month}.pgn.bz2")
    
    return f"lichess_db_standard_rated_{year_month}.pgn"

if __name__ == "__main__":
    # Example usage
    pgn_files = [
        "sample_games.pgn",
        "more_games.pgn"
    ]
    
    output_dir = "training_data/"
    data_dir = run_data_pipeline(pgn_files, output_dir) 