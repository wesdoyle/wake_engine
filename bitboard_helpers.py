import string

import numpy as np


def make_empty_uint64_bitmap():
    """
    Returns a numpy uint64 zero value
    """
    return np.uint64(0)


def get_bitboard_as_bytes(bitboard):
    return bitboard.tobytes()


def get_binary_string(bitboard, board_squares=64):
    return format(bitboard, 'b').zfill(board_squares)


# -------------------------------------------------------------
# BIT MANIPULATION
# -------------------------------------------------------------

def set_bit(bitboard, bit):
    return bitboard | np.uint64(1) << np.uint64(bit)


def clear_bit(bitboard, bit):
    return bitboard & ~(1 << bit)


# -------------------------------------------------------------
# DEBUG PRETTY PRINT
# -------------------------------------------------------------

def pprint_bb(bitboard, board_size=8):
    bitboard = get_binary_string(bitboard)
    val = ''
    display_rank = board_size
    board = [bitboard[i:i + 8] for i in range(0, len(bitboard), board_size)]
    for i, row in enumerate(board):
        val += f'{display_rank} '
        display_rank -= 1
        for square in reversed(row):
            if int(square):
                val += ' ▓'
                continue
            val += ' ░'
        val += '\n'
    val += '  '
    for char in string.ascii_uppercase[:board_size]:
        val += f' {char}'
    print(val)


# -------------------------------------------------------------
#  KNIGHT MOVEMENTS
# -------------------------------------------------------------

def generate_knight_attack_bb_from_square(square):
    bitmap = make_empty_uint64_bitmap()
    for i in [0, 6, 15, 17, 10, -6, -15, -17, -10]:
        bitmap |= set_bit(bitmap, square + i)
    return bitmap
