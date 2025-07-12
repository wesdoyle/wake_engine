"""
Bitboard package for the Wake chess engine.

This package provides a comprehensive set of bitboard operations for chess
programming, including core operations, ray generation, attack patterns,
static lookup tables, debug utilities, and board region access.

The package is organized into several modules:
- core: Basic bitboard operations and bit manipulation
- rays: Ray generation for sliding pieces
- attacks: Attack pattern generation for all piece types
- maps: Static lookup tables for O(1) attack pattern access
- debug: Visualization and debugging utilities
- regions: Board region access (ranks, files, special regions)

All functions are re-exported from this __init__.py for easy access.
"""

# Core bitboard operations
from .core import (
    BOARD_SIZE,
    BOARD_SQUARES,
    make_uint64,
    get_bitboard_as_bytes,
    get_binary_string,
    get_squares_from_bitboard,
    bitscan_forward,
    bitscan_reverse,
    set_bit,
    clear_bit,
)

# Ray generation functions
from .rays import (
    get_south_ray,
    get_north_ray,
    get_west_ray,
    get_east_ray,
    get_southeast_ray,
    get_northwest_ray,
    get_southwest_ray,
    get_northeast_ray,
)

# Attack pattern generation
from .attacks import (
    generate_knight_attack_bb_from_square,
    generate_rank_attack_bb_from_square,
    generate_file_attack_bb_from_square,
    generate_rook_attack_bb_from_square,
    generate_diag_attack_bb_from_square,
    generate_queen_attack_bb_from_square,
    generate_king_attack_bb_from_square,
    generate_white_pawn_attack_bb_from_square,
    generate_black_pawn_attack_bb_from_square,
    generate_white_pawn_motion_bb_from_square,
    generate_black_pawn_motion_bb_from_square,
)

# Static attack map builders
from .maps import (
    make_knight_attack_bbs,
    make_rank_attack_bbs,
    make_file_attack_bbs,
    make_diag_attack_bbs,
    make_king_attack_bbs,
    make_queen_attack_bbs,
    make_rook_attack_bbs,
    make_white_pawn_attack_bbs,
    make_black_pawn_attack_bbs,
    make_white_pawn_motion_bbs,
    make_black_pawn_motion_bbs,
)

# Debug and visualization
from .debug import (
    pprint_bb,
    pprint_pieces,
)

# Board region access
from .regions import (
    rank_8_bb,
    rank_7_bb,
    rank_6_bb,
    rank_5_bb,
    rank_4_bb,
    rank_3_bb,
    rank_2_bb,
    rank_1_bb,
    file_h_bb,
    file_g_bb,
    file_f_bb,
    file_e_bb,
    file_d_bb,
    file_c_bb,
    file_b_bb,
    file_a_bb,
    dark_squares_bb,
    light_squares_bb,
    center_squares_bb,
    flanks_bb,
    center_files_bb,
    kingside_bb,
    queenside_bb,
)

# Re-export all functions in __all__ for explicit public API
__all__ = [
    # Core
    "BOARD_SIZE",
    "BOARD_SQUARES",
    "make_uint64",
    "get_bitboard_as_bytes",
    "get_binary_string",
    "get_squares_from_bitboard",
    "bitscan_forward",
    "bitscan_reverse",
    "set_bit",
    "clear_bit",
    # Rays
    "get_south_ray",
    "get_north_ray",
    "get_west_ray",
    "get_east_ray",
    "get_southeast_ray",
    "get_northwest_ray",
    "get_southwest_ray",
    "get_northeast_ray",
    # Attacks
    "generate_knight_attack_bb_from_square",
    "generate_rank_attack_bb_from_square",
    "generate_file_attack_bb_from_square",
    "generate_rook_attack_bb_from_square",
    "generate_diag_attack_bb_from_square",
    "generate_queen_attack_bb_from_square",
    "generate_king_attack_bb_from_square",
    "generate_white_pawn_attack_bb_from_square",
    "generate_black_pawn_attack_bb_from_square",
    "generate_white_pawn_motion_bb_from_square",
    "generate_black_pawn_motion_bb_from_square",
    # Maps
    "make_knight_attack_bbs",
    "make_rank_attack_bbs",
    "make_file_attack_bbs",
    "make_diag_attack_bbs",
    "make_king_attack_bbs",
    "make_queen_attack_bbs",
    "make_rook_attack_bbs",
    "make_white_pawn_attack_bbs",
    "make_black_pawn_attack_bbs",
    "make_white_pawn_motion_bbs",
    "make_black_pawn_motion_bbs",
    # Debug
    "pprint_bb",
    "pprint_pieces",
    # Regions
    "rank_8_bb",
    "rank_7_bb",
    "rank_6_bb",
    "rank_5_bb",
    "rank_4_bb",
    "rank_3_bb",
    "rank_2_bb",
    "rank_1_bb",
    "file_h_bb",
    "file_g_bb",
    "file_f_bb",
    "file_e_bb",
    "file_d_bb",
    "file_c_bb",
    "file_b_bb",
    "file_a_bb",
    "dark_squares_bb",
    "light_squares_bb",
    "center_squares_bb",
    "flanks_bb",
    "center_files_bb",
    "kingside_bb",
    "queenside_bb",
]
