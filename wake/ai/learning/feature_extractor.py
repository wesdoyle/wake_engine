"""
Feature extraction framework for learning piece-square table values from game data.
This module defines the features and methods needed to build data-driven heuristics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from wake.constants import Piece, Color

@dataclass
class GameContext:
    """Context information for a position within a game"""
    move_number: int
    game_phase: str  # 'opening', 'middlegame', 'endgame'
    material_balance: float
    game_result: float  # 1.0 = white win, 0.5 = draw, 0.0 = black win
    player_ratings: Tuple[int, int]  # (white_elo, black_elo)
    time_control: str

@dataclass
class PositionFeatures:
    """Comprehensive features extracted from a chess position"""
    # Basic piece-square features
    piece_positions: Dict[int, List[int]]  # piece_type -> [squares]
    
    # Positional features
    king_safety_white: float
    king_safety_black: float
    pawn_structure_score: float
    piece_mobility: Dict[int, int]  # piece_type -> mobility_count
    
    # Control and influence
    square_control: np.ndarray  # 64-element array of square control values
    central_control: float
    
    # Game phase indicators
    material_count: Dict[int, int]
    game_phase_score: float  # 0.0 = endgame, 1.0 = opening
    
    # Tactical features
    pins_and_skewers: int
    hanging_pieces: int
    piece_coordination: float

class FeatureExtractor:
    """Extracts features from chess positions for learning"""
    
    def __init__(self):
        self.piece_values = {
            Piece.wP: 1, Piece.bP: 1,
            Piece.wN: 3, Piece.bN: 3,
            Piece.wB: 3, Piece.bB: 3,
            Piece.wR: 5, Piece.bR: 5,
            Piece.wQ: 9, Piece.bQ: 9,
        }
    
    def extract_position_features(self, position, game_context: GameContext) -> PositionFeatures:
        """Extract comprehensive features from a position"""
        return PositionFeatures(
            piece_positions=self._extract_piece_positions(position),
            king_safety_white=self._calculate_king_safety(position, Color.WHITE),
            king_safety_black=self._calculate_king_safety(position, Color.BLACK),
            pawn_structure_score=self._evaluate_pawn_structure(position),
            piece_mobility=self._calculate_piece_mobility(position),
            square_control=self._calculate_square_control(position),
            central_control=self._calculate_central_control(position),
            material_count=self._count_material(position),
            game_phase_score=self._calculate_game_phase(position, game_context),
            pins_and_skewers=self._count_tactical_motifs(position),
            hanging_pieces=self._count_hanging_pieces(position),
            piece_coordination=self._calculate_piece_coordination(position)
        )
    
    def create_training_sample(self, position, game_context: GameContext) -> Dict:
        """Create a training sample for machine learning"""
        features = self.extract_position_features(position, game_context)
        
        # Create feature vector for ML
        feature_vector = self._features_to_vector(features)
        
        # Weight the outcome by rating difference and game phase
        outcome_weight = self._calculate_outcome_weight(game_context)
        
        return {
            'features': feature_vector,
            'outcome': game_context.game_result,
            'weight': outcome_weight,
            'game_phase': features.game_phase_score,
            'position_id': self._generate_position_id(position)
        }
    
    def _extract_piece_positions(self, position) -> Dict[int, List[int]]:
        """Extract piece positions as square indices"""
        piece_positions = {}
        for piece_type, squares in position.piece_map.items():
            piece_positions[piece_type] = list(squares)
        return piece_positions
    
    def _calculate_king_safety(self, position, color: Color) -> float:
        """Calculate king safety score"""
        # TODO: Implement king safety calculation
        # Consider: pawn shield, piece proximity, attack maps
        return 0.0
    
    def _evaluate_pawn_structure(self, position) -> float:
        """Evaluate pawn structure quality"""
        # TODO: Implement pawn structure evaluation
        # Consider: doubled, isolated, backward pawns, pawn chains
        return 0.0
    
    def _calculate_piece_mobility(self, position) -> Dict[int, int]:
        """Calculate mobility for each piece type"""
        mobility = {}
        for piece_type in [Piece.wN, Piece.bN, Piece.wB, Piece.bB, 
                          Piece.wR, Piece.bR, Piece.wQ, Piece.bQ]:
            if piece_type in position.piece_map:
                # Count legal moves for pieces of this type
                mobility[piece_type] = len(position.piece_map[piece_type]) * 10  # Placeholder
        return mobility
    
    def _calculate_square_control(self, position) -> np.ndarray:
        """Calculate control value for each square"""
        control = np.zeros(64)
        # TODO: Implement square control calculation
        # Use attack bitboards to determine square control
        return control
    
    def _calculate_central_control(self, position) -> float:
        """Calculate control of central squares (d4, d5, e4, e5)"""
        central_squares = [27, 28, 35, 36]  # d4, e4, d5, e5
        control = self._calculate_square_control(position)
        return np.sum(control[central_squares])
    
    def _count_material(self, position) -> Dict[int, int]:
        """Count pieces by type"""
        material = {}
        for piece_type, squares in position.piece_map.items():
            material[piece_type] = len(squares)
        return material
    
    def _calculate_game_phase(self, position, game_context: GameContext) -> float:
        """Calculate game phase score (0.0 = endgame, 1.0 = opening)"""
        material = self._count_material(position)
        total_material = sum(
            count * self.piece_values.get(piece_type, 0)
            for piece_type, count in material.items()
        )
        # Normalize based on starting material (78 points total)
        return min(1.0, total_material / 78.0)
    
    def _count_tactical_motifs(self, position) -> int:
        """Count pins, skewers, and other tactical motifs"""
        # TODO: Implement tactical motif detection
        return 0
    
    def _count_hanging_pieces(self, position) -> int:
        """Count undefended pieces under attack"""
        # TODO: Implement hanging piece detection
        return 0
    
    def _calculate_piece_coordination(self, position) -> float:
        """Calculate how well pieces work together"""
        # TODO: Implement piece coordination scoring
        return 0.0
    
    def _features_to_vector(self, features: PositionFeatures) -> np.ndarray:
        """Convert feature object to numpy vector for ML"""
        vector_parts = []
        
        # Piece position encoding (one-hot or similar)
        for piece_type in [Piece.wP, Piece.wN, Piece.wB, Piece.wR, Piece.wQ, Piece.wK,
                          Piece.bP, Piece.bN, Piece.bB, Piece.bR, Piece.bQ, Piece.bK]:
            piece_vector = np.zeros(64)
            if piece_type in features.piece_positions:
                for square in features.piece_positions[piece_type]:
                    piece_vector[square] = 1.0
            vector_parts.append(piece_vector)
        
        # Add other features
        vector_parts.extend([
            np.array([features.king_safety_white]),
            np.array([features.king_safety_black]),
            np.array([features.pawn_structure_score]),
            np.array([features.central_control]),
            np.array([features.game_phase_score]),
            features.square_control
        ])
        
        return np.concatenate(vector_parts)
    
    def _calculate_outcome_weight(self, game_context: GameContext) -> float:
        """Calculate weight for this position based on game context"""
        # Weight by rating level (higher rated games = more weight)
        avg_rating = sum(game_context.player_ratings) / 2
        rating_weight = min(2.0, avg_rating / 2000.0)
        
        # Weight by game phase (critical middlegame positions = more weight)
        phase_weight = 1.0 + 0.5 * (1.0 - abs(0.5 - (game_context.move_number / 80.0)))
        
        return rating_weight * phase_weight
    
    def _generate_position_id(self, position) -> str:
        """Generate unique ID for position (could use FEN or hash)"""
        # TODO: Implement position hashing
        return "pos_" + str(hash(str(position.piece_map)))

# Usage example:
"""
extractor = FeatureExtractor()

# For each position in training data:
game_context = GameContext(
    move_number=20,
    game_phase='middlegame', 
    material_balance=0.5,
    game_result=1.0,  # White won
    player_ratings=(2200, 2180),
    time_control='blitz'
)

training_sample = extractor.create_training_sample(position, game_context)
""" 