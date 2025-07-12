"""
Ray generation functions for sliding pieces.

This module provides functions to generate rays in all 8 directions
for sliding pieces (rooks, bishops, queens) on an otherwise empty board.
"""

import numpy as np

from wake.constants import File, HOT
from .core import clear_bit


def get_south_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of south sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the south ray sliding attacks from `square`
    :param from_square: The square from a south-sliding piece attacks
    :return: np.uint64 bitboard of the southern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    while 0 <= from_square < 64:
        bitboard |= HOT << np.uint64(from_square)
        if from_square < 8:  # Can't go further south
            break
        from_square -= 8

    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_north_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of north sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the north ray sliding attacks from `square`
    :param from_square: The square from a north-sliding piece attacks
    :return: np.uint64 bitboard of the northern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    while 0 <= from_square < 64:
        bitboard |= HOT << np.uint64(from_square)
        if from_square >= 56:  # Can't go further north (rank 8)
            break
        from_square += 8

    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_west_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of west sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the west ray sliding attacks from `square`
    :param from_square: The square from a west-sliding piece attacks
    :return: np.uint64 bitboard of the western squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    while 0 <= from_square < 64:
        bitboard |= HOT << np.uint64(from_square)
        if from_square % 8 == 0:
            break
        from_square -= 1

    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_east_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of east sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the east ray sliding attacks from `square`
    :param from_square: The square from a east-sliding piece attacks
    :return: np.uint64 bitboard of the eastern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    while 0 <= from_square < 64:
        bitboard |= HOT << np.uint64(from_square)
        if from_square % 8 == 7:  # H file - can't go further east
            break
        from_square += 1

    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_southeast_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of southeast sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the southeast ray sliding attacks from `square`
    :param from_square: The square from a southeast-sliding piece attacks
    :return: np.uint64 bitboard of the southeastern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    while 0 <= from_square < 64:
        bitboard |= HOT << np.uint64(from_square)
        if from_square % 8 == 7 or from_square < 8:  # H file or rank 1
            break
        from_square -= 7

    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_northwest_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of northwest sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the northwest ray sliding attacks from `square`
    :param from_square: The square from a northwest-sliding piece attacks
    :return: np.uint64 bitboard of the northwestern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    while 0 <= from_square < 64:
        bitboard |= HOT << np.uint64(from_square)
        if from_square % 8 == 0 or from_square >= 56:  # A file or rank 8
            break
        from_square += 7

    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_southwest_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of southwest sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the southwest ray sliding attacks from `square`
    :param from_square: The square from a southwest-sliding piece attacks
    :return: np.uint64 bitboard of the southwestern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    while 0 <= from_square < 64:
        bitboard |= HOT << np.uint64(from_square)
        if from_square % 8 == 0 or from_square < 8:  # A file or rank 1
            break
        from_square -= 9

    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_northeast_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of northeast sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the northeast ray sliding attacks from `square`
    :param from_square: The square from a northeast-sliding piece attacks
    :return: np.uint64 bitboard of the northeastern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    while 0 <= from_square < 64:
        bitboard |= HOT << np.uint64(from_square)
        if from_square % 8 == 7 or from_square >= 56:  # H file or rank 8
            break
        from_square += 9

    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard
