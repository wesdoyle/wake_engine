"""
Debug and visualization functions for bitboards.

This module provides utilities for debugging and visualizing bitboard
states, including pretty-printing bitboards and piece positions.
"""

import string
import numpy as np

from wake.constants import piece_to_glyph
from .core import get_binary_string


def pprint_bb(bitboard: np.uint64, board_size: int = 8) -> None:
    """
    Pretty-prints the given bitboard as 8 x 8 chess board
    :param bitboard: the bitboard to pretty-print
    :param board_size: the length of the square board
    :return: None
    """
    bitboard = get_binary_string(bitboard)
    val = ""
    display_rank = board_size
    board = [bitboard[i : i + 8] for i in range(0, len(bitboard), board_size)]
    for i, row in enumerate(board):
        val += f"{display_rank} "
        display_rank -= 1
        for square in reversed(row):
            if int(square):
                val += " ▓"
                continue
            val += " ░"
        val += "\n"
    val += "  "
    for char in string.ascii_uppercase[:board_size]:
        val += f" {char}"
    print(val)


def pprint_pieces(piece_map: dict, board_size: int = 8) -> None:
    """
    Prints the given piece map as 8 x 8 chess board using Unicode chess symbols
    :param piece_map: Python dictionary of piece to set of square indices
    :param board_size: the length of the square board
    :return: None
    """
    board = ["░"] * 64
    for piece, squares in piece_map.items():
        for square in squares:
            board[square] = piece_to_glyph[piece]
    board = np.array(board)
    board = np.reshape(board, (8, 8))
    display_rank = board_size
    for i, row in enumerate(reversed(board)):
        res = f"{display_rank} "
        display_rank -= 1
        for glyph in row:
            res += f" {glyph}"
        print(res)
    res = "  "
    for char in string.ascii_uppercase[:board_size]:
        res += f" {char}"
    print(res)
