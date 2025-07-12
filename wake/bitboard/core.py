"""
Core bitboard operations for the Wake chess engine.

This module provides fundamental bitboard operations including:
- Basic bitboard creation and manipulation
- Bit querying (forward/reverse scanning)
- Bit manipulation (set/clear operations)
- Utility functions for bitboard representation
"""

import numpy as np

# Constants
BOARD_SIZE = 8
BOARD_SQUARES = BOARD_SIZE**2


def make_uint64() -> np.uint64:
    """
    :return: an np.uint64 zero
    """
    return np.uint64(0)


def get_bitboard_as_bytes(bitboard: np.uint64) -> bytes:
    """
    Returns the provided bitboard as Python bytes representation
    :param bitboard:
    :return:
    """
    return bitboard.tobytes()


def get_binary_string(bitboard: np.uint64, board_squares: int = 64) -> str:
    """
    Returns the binary string representation of the provided bitboard
    :param bitboard: the bitboard to be represented
    :param board_squares: the number of squares in the bitboard
    :return: string representation of the provided n**2 (board_squares) bitboard
    """
    return format(bitboard, "b").zfill(board_squares)


def get_squares_from_bitboard(bitboard: np.uint64) -> list:
    """
    Returns a list of square indices where bits are set in the bitboard
    :param bitboard: the bitboard to analyze
    :return: list of square indices (0-63) where bits are set
    """
    squares = []
    temp = np.uint64(bitboard)

    # Efficient bit manipulation to find all set bits
    index = 0
    while temp != 0:
        if temp & 1:
            squares.append(index)
        temp >>= 1
        index += 1

    return squares


# -------------------------------------------------------------
# BIT QUERYING
# -------------------------------------------------------------


def bitscan_forward(bitboard: np.uint64) -> int:
    """
    Returns the least significant one bit from the provided bitboard
    :param bitboard: bitboard to scan
    :return: int significant one bit binary string index (1-based)
    """
    if bitboard == 0:
        return -1  # No bits set

    # Use efficient bit manipulation to find the index
    index = 0
    temp = np.uint64(bitboard)

    # Binary search approach for O(log n) complexity
    if temp & np.uint64(0xFFFFFFFF) == 0:
        index += 32
        temp >>= 32
    if temp & np.uint64(0xFFFF) == 0:
        index += 16
        temp >>= 16
    if temp & np.uint64(0xFF) == 0:
        index += 8
        temp >>= 8
    if temp & np.uint64(0xF) == 0:
        index += 4
        temp >>= 4
    if temp & np.uint64(0x3) == 0:
        index += 2
        temp >>= 2
    if temp & np.uint64(0x1) == 0:
        index += 1

    return index + 1  # Return 1-based index to match original API


def bitscan_reverse(bitboard: np.uint64) -> np.uint64 or int:
    """
    @author Eugene Nalimov
    @return index (0..63) of most significant one bit
    :param bitboard: bitboard to scan
    :return: np.uint64 most significant one bit binary string index
    """

    def lookup_most_significant_1_bit(bit: np.uint64) -> int:
        if bit > np.uint64(127):
            return np.uint64(7)
        if bit > np.uint64(63):
            return np.uint64(6)
        if bit > np.uint64(31):
            return np.uint64(5)
        if bit > np.uint64(15):
            return np.uint64(4)
        if bit > np.uint64(7):
            return np.uint64(3)
        if bit > np.uint64(3):
            return np.uint64(2)
        if bit > np.uint64(1):
            return np.uint64(1)
        return np.uint64(0)

    if not bitboard:
        raise Exception("You don't want to reverse scan en empty bitboard, right?")

    result = np.uint64(0)

    if bitboard > 0xFFFFFFFF:
        bitboard >>= np.uint(32)
        result = np.uint(32)

    if bitboard > 0xFFFF:
        bitboard >>= np.uint(16)
        result += np.uint(16)

    if bitboard > 0xFF:
        bitboard >>= np.uint(8)
        result += np.uint(8)

    return result + lookup_most_significant_1_bit(bitboard)


# -------------------------------------------------------------
# BIT MANIPULATION
# -------------------------------------------------------------


def set_bit(bitboard: np.uint64, bit: int) -> np.uint64:
    """
    Sets a bit in the provided unsigned 64-bit integer bitboard representation to 1
    :param bitboard: np.uint64 number
    :param bit: the binary index to turn hot
    :return: a copy of the bitboard with the specified `bit` set to 1
    """
    return np.uint64(bitboard | np.uint64(1) << np.uint64(bit))


def clear_bit(bitboard: np.uint64, bit: int or np.uint64) -> np.uint64:
    """
    Sets a bit in the provided unsigned 64-bit integer bitboard representation to 0
    :param bitboard: np.uint64 number
    :param bit: the binary index to turn off
    :return: a copy of the bitboard with the specified `bit` set to 0
    """
    return bitboard & ~(np.uint64(1) << np.uint64(bit))
