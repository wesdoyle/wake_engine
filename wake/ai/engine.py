"""
AI Engine Interface for Wake Chess Engine

This module provides the main AIEngine class that serves as the interface
between the chess game and the AI search algorithms.
"""

import time
from typing import Optional, Dict, Any
from wake.constants import Color
from wake.move import Move
from .search import search_to_depth, SearchResult
from .evaluation import evaluate_position, format_evaluation


class AIEngine:
    """
    Main AI Engine class that provides an interface for chess AI functionality.
    
    This class manages AI settings, search parameters, and provides methods
    for the chess game to get the best move from any position.
    """
    
    def __init__(self, default_depth: int = 4, use_alpha_beta: bool = True):
        """
        Initialize the AI Engine.
        
        Args:
            default_depth: Default search depth in plies
            use_alpha_beta: Whether to use alpha-beta pruning (recommended)
        """
        self.default_depth = default_depth
        self.use_alpha_beta = use_alpha_beta
        self.debug = False
        
        # Search statistics
        self.stats = {
            'total_searches': 0,
            'total_nodes_searched': 0,
            'total_time_spent': 0.0,
            'average_nodes_per_second': 0.0
        }
    
    def get_best_move(self, position, depth: Optional[int] = None, 
                     time_limit: Optional[float] = None) -> SearchResult:
        """
        Get the best move for the current position.
        
        Args:
            position: Current chess position
            depth: Search depth (uses default if None)
            time_limit: Time limit in seconds (not implemented yet)
        
        Returns:
            SearchResult containing the best move and analysis
        """
        if depth is None:
            depth = self.default_depth
        
        # Perform the search
        result = search_to_depth(
            position=position,
            target_depth=depth,
            use_alpha_beta=self.use_alpha_beta,
            debug=self.debug
        )
        
        # Update statistics
        self._update_stats(result)
        
        return result
    
    def evaluate_current_position(self, position) -> Dict[str, Any]:
        """
        Evaluate the current position without searching.
        
        Args:
            position: Chess position to evaluate
        
        Returns:
            Dictionary with evaluation details
        """
        evaluation = evaluate_position(position)
        legal_moves = position.generate_legal_moves()
        
        return {
            'evaluation': evaluation,
            'evaluation_string': format_evaluation(evaluation),
            'legal_moves_count': len(legal_moves),
            'color_to_move': 'White' if position.color_to_move == Color.WHITE else 'Black',
            'in_check': bool(position.king_in_check[position.color_to_move])
        }
    
    def analyze_position(self, position, depth: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform a deep analysis of the position.
        
        Args:
            position: Chess position to analyze
            depth: Analysis depth (uses default if None)
        
        Returns:
            Dictionary with detailed analysis
        """
        # Get basic evaluation
        basic_eval = self.evaluate_current_position(position)
        
        # Get best move through search
        search_result = self.get_best_move(position, depth)
        
        return {
            'position_evaluation': basic_eval,
            'search_result': search_result,
            'best_move': search_result.best_move,
            'search_evaluation': search_result.evaluation,
            'search_evaluation_string': format_evaluation(search_result.evaluation),
            'depth_searched': search_result.depth,
            'nodes_searched': search_result.nodes_searched,
            'search_time': search_result.time_taken,
            'nodes_per_second': (search_result.nodes_searched / search_result.time_taken) 
                                if search_result.time_taken > 0 else 0
        }
    
    def suggest_move(self, position, depth: Optional[int] = None) -> Move:
        """
        Get a move suggestion for the current position.
        
        Args:
            position: Current chess position
            depth: Search depth (uses default if None)
        
        Returns:
            Best move found, or None if no legal moves
        """
        result = self.get_best_move(position, depth)
        return result.best_move
    
    def set_strength(self, depth: int, use_alpha_beta: bool = True):
        """
        Set AI strength by adjusting search parameters.
        
        Args:
            depth: Search depth (1=very weak, 6+=strong)
            use_alpha_beta: Whether to use alpha-beta pruning
        """
        self.default_depth = max(1, min(depth, 10))  # Clamp between 1-10
        self.use_alpha_beta = use_alpha_beta
        
        if self.debug:
            print(f"AI strength set to depth {self.default_depth}, "
                  f"alpha-beta {'enabled' if use_alpha_beta else 'disabled'}")
    
    def enable_debug(self, enabled: bool = True):
        """Enable or disable debug output."""
        self.debug = enabled
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get AI engine statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset AI engine statistics."""
        self.stats = {
            'total_searches': 0,
            'total_nodes_searched': 0,
            'total_time_spent': 0.0,
            'average_nodes_per_second': 0.0
        }
    
    def _update_stats(self, result: SearchResult):
        """Update internal statistics with search result."""
        self.stats['total_searches'] += 1
        self.stats['total_nodes_searched'] += result.nodes_searched
        self.stats['total_time_spent'] += result.time_taken
        
        if self.stats['total_time_spent'] > 0:
            self.stats['average_nodes_per_second'] = (
                self.stats['total_nodes_searched'] / self.stats['total_time_spent']
            )
    
    def __str__(self):
        """String representation of the AI engine."""
        strength = "weak" if self.default_depth <= 2 else "medium" if self.default_depth <= 4 else "strong"
        algorithm = "Alpha-Beta" if self.use_alpha_beta else "Minimax"
        
        return f"AIEngine(depth={self.default_depth}, algorithm={algorithm}, strength={strength})"


# Convenience functions for quick AI usage
def get_ai_move(position, depth: int = 4) -> Move:
    """
    Quick function to get an AI move.
    
    Args:
        position: Current chess position
        depth: Search depth
    
    Returns:
        Best move found
    """
    engine = AIEngine(default_depth=depth)
    return engine.suggest_move(position)


def evaluate_position_quick(position) -> str:
    """
    Quick function to evaluate a position.
    
    Args:
        position: Chess position to evaluate
    
    Returns:
        Formatted evaluation string
    """
    evaluation = evaluate_position(position)
    return format_evaluation(evaluation) 