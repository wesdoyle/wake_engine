"""
Position Evaluation for Wake Chess Engine

This module provides functions to evaluate chess positions numerically.
Positive values favor White, negative values favor Black.
"""
from wake.ai.constants.piece_square_tables import PIECE_TABLES
from wake.ai.constants.piece_values import PIECE_VALUES
from wake.constants import Piece, Color

def material_value(position) -> int:
    """
    Calculate the material value of the position.
    Returns positive value if White has material advantage.
    """
    total = 0
    for piece, squares in position.piece_map.items():
        if piece in PIECE_VALUES:
            piece_count = len(squares)
            total += PIECE_VALUES[piece] * piece_count
    return total


def positional_value(position) -> int:
    """
    Calculate positional value based on piece-square tables.
    Returns positive value if White has positional advantage.
    """
    total = 0
    
    for piece, squares in position.piece_map.items():
        if piece in PIECE_TABLES:
            table = PIECE_TABLES[piece]
            for square in squares:
                if piece in Piece.white_pieces:
                    # White pieces use table as-is
                    total += table[square]
                else:
                    # Black pieces use flipped table and negative value
                    flipped_square = 63 - square  # Flip the board
                    white_piece = piece - 6  # Convert black piece to white equivalent
                    if white_piece in PIECE_TABLES:
                        total -= PIECE_TABLES[white_piece][flipped_square]
    
    return total


def evaluate_position(position) -> int:
    """
    Main position evaluation function.
    
    Returns:
        int: Position evaluation in centipawns
             Positive = White advantage
             Negative = Black advantage
             0 = Equal position
    """
    # Check for terminal positions first
    legal_moves = position.generate_legal_moves()
    
    if len(legal_moves) == 0:
        # No legal moves - checkmate or stalemate
        if position.king_in_check[position.color_to_move]:
            # Checkmate - very bad for side to move
            return -999999 if position.color_to_move == Color.WHITE else 999999
        else:
            # Stalemate - draw
            return 0
    
    # calculate material and positional values
    material = material_value(position)
    positional = positional_value(position)
    
    # basic mobility bonus (having more legal moves is generally good)
    mobility = len(legal_moves) * 2
    if position.color_to_move == Color.BLACK:
        mobility = -mobility
    
    # Combine all factors
    total_evaluation = material + positional + mobility
    
    return total_evaluation


def format_evaluation(centipawns: int) -> str:
    """
    Format evaluation for display.
    
    Args:
        centipawns: Evaluation in centipawns
    
    Returns:
        Formatted string (e.g., "+1.25", "-0.50", "0.00")
    """
    pawns = centipawns / 100.0
    return f"{pawns:+.2f}" 