"""
Wake Chess Engine AI Module

This module contains all AI-related functionality including:
- Position evaluation
- Search algorithms (minimax, alpha-beta)
- Main AI engine interface
"""

from .evaluation import evaluate_position, material_value
from .search import minimax, alpha_beta
from .engine import AIEngine, get_ai_move, evaluate_position_quick

__all__ = [
    'evaluate_position',
    'material_value', 
    'minimax',
    'alpha_beta',
    'AIEngine',
    'get_ai_move',
    'evaluate_position_quick'
] 