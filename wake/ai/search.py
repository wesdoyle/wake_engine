"""
Search Algorithms for Wake Chess Engine

This module implements tree search algorithms including:
- Minimax
- Alpha-Beta pruning
- Move ordering
"""

import time
from typing import Tuple, Optional, List
from wake.constants import Color
from wake.move import Move
from .evaluation import evaluate_position


class SearchResult:
    """Container for search results"""
    def __init__(self, best_move: Optional[Move] = None, evaluation: int = 0, 
                 depth: int = 0, nodes_searched: int = 0, time_taken: float = 0.0):
        self.best_move = best_move
        self.evaluation = evaluation  # In centipawns
        self.depth = depth
        self.nodes_searched = nodes_searched
        self.time_taken = time_taken
    
    def __str__(self):
        eval_str = f"{self.evaluation/100:+.2f}"
        return (f"Best: {self.best_move} | Eval: {eval_str} | "
                f"Depth: {self.depth} | Nodes: {self.nodes_searched:,} | "
                f"Time: {self.time_taken:.3f}s")


def minimax(position, depth: int, maximizing_player: bool, 
            debug: bool = False) -> SearchResult:
    """
    Basic minimax search algorithm.
    
    Args:
        position: Current chess position
        depth: Search depth (number of plies)
        maximizing_player: True if maximizing (White), False if minimizing (Black)
        debug: Print debug information
    
    Returns:
        SearchResult with best move and evaluation
    """
    start_time = time.time()
    nodes_searched = 0
    
    def minimax_recursive(pos, depth: int, maximizing: bool) -> Tuple[int, Optional[Move]]:
        nonlocal nodes_searched
        nodes_searched += 1
        
        # Base case: leaf node
        if depth == 0:
            return evaluate_position(pos), None
        
        # Get all legal moves
        legal_moves = pos.generate_legal_moves()
        
        # Terminal position check
        if len(legal_moves) == 0:
            return evaluate_position(pos), None
        
        best_move = None
        
        if maximizing:
            # Maximizing player (White)
            max_eval = float('-inf')
            
            for move in legal_moves:
                # Save position state
                move_state = pos.save_move_state(move)
                
                # Make move
                pos.make_move_fast(move)
                
                # Recursive search
                evaluation, _ = minimax_recursive(pos, depth - 1, False)
                
                # Unmake move
                pos.unmake_move_fast(move, move_state)
                
                # Update best move
                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move
            
            return max_eval, best_move
        
        else:
            # Minimizing player (Black)
            min_eval = float('inf')
            
            for move in legal_moves:
                # Save position state
                move_state = pos.save_move_state(move)
                
                # Make move
                pos.make_move_fast(move)
                
                # Recursive search
                evaluation, _ = minimax_recursive(pos, depth - 1, True)
                
                # Unmake move
                pos.unmake_move_fast(move, move_state)
                
                # Update best move
                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = move
            
            return min_eval, best_move
    
    # Perform search
    evaluation, best_move = minimax_recursive(position, depth, maximizing_player)
    
    # Calculate time taken
    time_taken = time.time() - start_time
    
    if debug:
        print(f"Minimax search completed: depth={depth}, nodes={nodes_searched:,}, time={time_taken:.3f}s")
    
    return SearchResult(
        best_move=best_move,
        evaluation=evaluation,
        depth=depth,
        nodes_searched=nodes_searched,
        time_taken=time_taken
    )


def alpha_beta(position, depth: int, maximizing_player: bool, 
               alpha: int = float('-inf'), beta: int = float('inf'),
               debug: bool = False) -> SearchResult:
    """
    Alpha-Beta pruning search algorithm.
    
    Args:
        position: Current chess position
        depth: Search depth (number of plies)
        maximizing_player: True if maximizing (White), False if minimizing (Black)
        alpha: Alpha value for pruning
        beta: Beta value for pruning
        debug: Print debug information
    
    Returns:
        SearchResult with best move and evaluation
    """
    start_time = time.time()
    nodes_searched = 0
    pruned_branches = 0
    
    def alpha_beta_recursive(pos, depth: int, alpha: int, beta: int, 
                           maximizing: bool) -> Tuple[int, Optional[Move]]:
        nonlocal nodes_searched, pruned_branches
        nodes_searched += 1
        
        # Base case: leaf node
        if depth == 0:
            return evaluate_position(pos), None
        
        # Get all legal moves
        legal_moves = pos.generate_legal_moves()
        
        # Terminal position check
        if len(legal_moves) == 0:
            return evaluate_position(pos), None
        
        # Order moves for better pruning (captures first, checks, etc.)
        ordered_moves = order_moves(legal_moves, pos)
        
        best_move = None
        
        if maximizing:
            # Maximizing player (White)
            max_eval = float('-inf')
            
            for move in ordered_moves:
                # Save position state
                move_state = pos.save_move_state(move)
                

                # Make move
                pos.make_move_fast(move)
                
                # Recursive search
                evaluation, _ = alpha_beta_recursive(pos, depth - 1, alpha, beta, False)
                
                # Unmake move
                pos.unmake_move_fast(move, move_state)
                
                # Update best move
                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move
                
                # Alpha-Beta pruning
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    pruned_branches += 1
                    break  # Beta cutoff
            
            return max_eval, best_move
        
        else:
            # Minimizing player (Black)
            min_eval = float('inf')
            
            for move in ordered_moves:
                # Save position state
                move_state = pos.save_move_state(move)
                

                # Make move
                pos.make_move_fast(move)
                
                # Recursive search
                evaluation, _ = alpha_beta_recursive(pos, depth - 1, alpha, beta, True)
                
                # Unmake move
                pos.unmake_move_fast(move, move_state)
                
                # Update best move
                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = move
                
                # Alpha-Beta pruning
                beta = min(beta, evaluation)
                if beta <= alpha:
                    pruned_branches += 1
                    break  # Alpha cutoff
            
            return min_eval, best_move
    
    # Perform search
    evaluation, best_move = alpha_beta_recursive(position, depth, alpha, beta, maximizing_player)
    
    # Calculate time taken
    time_taken = time.time() - start_time
    
    if debug:
        print(f"Alpha-Beta search completed: depth={depth}, nodes={nodes_searched:,}, "
              f"pruned={pruned_branches:,}, time={time_taken:.3f}s")
    
    return SearchResult(
        best_move=best_move,
        evaluation=evaluation,
        depth=depth,
        nodes_searched=nodes_searched,
        time_taken=time_taken
    )


def order_moves(moves: List[Move], position) -> List[Move]:
    """
    Order moves for better alpha-beta pruning using MVV-LVA and other heuristics.
    Good moves should be searched first to maximize pruning.
    
    Move ordering heuristics:
    1. MVV-LVA (Most Valuable Victim - Least Valuable Attacker) for captures
    2. Promotions (especially to queen)
    3. Checks
    4. Castling
    5. Other moves
    """
    # MVV-LVA piece values (victim value - attacker value)
    MVV_LVA_VALUES = {
        # Piece values for MVV-LVA (higher = more valuable)
        'wP': 100, 'bP': 100,
        'wN': 320, 'bN': 320,
        'wB': 330, 'bB': 330,
        'wR': 500, 'bR': 500,
        'wQ': 900, 'bQ': 900,
        'wK': 20000, 'bK': 20000,  # King is invaluable
    }
    
    def get_piece_value(piece):
        """Get MVV-LVA value for a piece"""
        if piece is None:
            return 0
        # Map piece constants to string keys
        from wake.constants import Piece
        piece_map = {
            Piece.wP: 'wP', Piece.bP: 'bP',
            Piece.wN: 'wN', Piece.bN: 'bN', 
            Piece.wB: 'wB', Piece.bB: 'bB',
            Piece.wR: 'wR', Piece.bR: 'bR',
            Piece.wQ: 'wQ', Piece.bQ: 'bQ',
            Piece.wK: 'wK', Piece.bK: 'bK',
        }
        return MVV_LVA_VALUES.get(piece_map.get(piece, ''), 0)
    
    def move_priority(move: Move) -> int:
        priority = 0
        
        # 1. MVV-LVA for captures
        if move.is_capture:
            target_piece = position.mailbox[move.to_sq] if move.to_sq < len(position.mailbox) else None
            victim_value = get_piece_value(target_piece)
            attacker_value = get_piece_value(move.piece)
            
            # MVV-LVA score: prioritize high-value victims captured by low-value attackers
            mvv_lva_score = victim_value * 10 - attacker_value
            priority += 10000 + mvv_lva_score
        
        # 2. Promotions (especially to queen)
        if move.is_promotion:
            priority += 9000
        
        # 3. Castling (generally good positional move)
        if move.is_castling:
            priority += 500
        
        # 4. TODO: Add check detection for move ordering
        
        return priority
    
    # Sort moves by priority (highest first)
    return sorted(moves, key=move_priority, reverse=True)


def search_to_depth(position, target_depth: int, use_alpha_beta: bool = True, 
                   debug: bool = False) -> SearchResult:
    """
    Search to a specific depth using the best available algorithm.
    
    Args:
        position: Current chess position
        target_depth: Target search depth
        use_alpha_beta: Use alpha-beta pruning if True, basic minimax if False
        debug: Print debug information
    
    Returns:
        SearchResult with best move and evaluation
    """
    maximizing = (position.color_to_move == Color.WHITE)
    
    if use_alpha_beta:
        return alpha_beta(position, target_depth, maximizing, debug=debug)
    else:
        return minimax(position, target_depth, maximizing, debug=debug) 