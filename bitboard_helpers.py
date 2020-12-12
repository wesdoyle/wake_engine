import string

import numpy as np

from constants import File


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
# BIT QUERYING
# -------------------------------------------------------------

def bitscan_forward(bitboard):
    """ Scans from A1 until we hit a hot bit """
    i = 1
    while not (bitboard >> np.uint64(i)) % 2:
        i += 1
    return i


def bitscan_reverse(bitboard):
    # bitScanReverse
    # @author Eugene Nalimov
    # @param bb bitboard to scan
    # @return index (0..63) of most significant one bit
    #
    def lookup_most_significant_1_bit(bit: np.uint64) -> np.uint64:
        if bit > np.uint64(127): return np.uint64(7)
        if bit > np.uint64(63):  return np.uint64(6)
        if bit > np.uint64(31):  return np.uint64(5)
        if bit > np.uint64(15):  return np.uint64(4)
        if bit > np.uint64(7):   return np.uint64(3)
        if bit > np.uint64(1):   return np.uint64(1)
        return np.uint64(0)

    if not bitboard:
        raise Exception("You don't want to reverse scan en empty bitboard, right?")

    result = np.uint64(0)

    if bitboard > 0xFFFFFFFF:
        bitboard >>= 32
        result = 32

    if bitboard > 0xFFFF:
        bitboard >>= 16
        result += 16

    if bitboard > 0xFF:
        bitboard >>= 8
        result += 8

    return result + lookup_most_significant_1_bit(bitboard)


# -------------------------------------------------------------
# BIT MANIPULATION
# -------------------------------------------------------------

def set_bit(bitboard, bit):
    return bitboard | np.uint64(1) << np.uint64(bit)


def clear_bit(bitboard, bit):
    return bitboard & ~(np.uint64(1) << bit)


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
#  ATTACK PATTERNS
# -------------------------------------------------------------

def generate_knight_attack_bb_from_square(square):
    attack_bb = make_empty_uint64_bitmap()
    for i in [0, 6, 15, 17, 10, -6, -15, -17, -10]:
        attack_bb |= set_bit(attack_bb, square + i)
    return attack_bb


def generate_rank_attack_bb_from_square(square):
    attack_bb = make_empty_uint64_bitmap()
    # North
    for i in range(0, 64, 8):
        attack_bb |= set_bit(square + i)
    # South
    for i in range(0, -64, -8):
        attack_bb |= set_bit(square + i)
    return attack_bb


def generate_file_attack_bb_from_square(square):
    attack_bb = make_empty_uint64_bitmap()
    hot = np.uint64(1)
    # East
    if not square % 8:
        attack_bb |= hot << np.uint64(square)
        square += 1
    while not square % 8 == 0:
        attack_bb |= hot << np.uint64(square)
        square += 1
    # West
    if square % 8 == 0:
        attack_bb |= hot << np.uint64(square)
        square -= 1
    else:
        while not square % 8 == 0:
            attack_bb |= hot << np.uint64(square)
            square -= 1
        attack_bb |= hot << np.uint64(square)
    return attack_bb


def generate_diag_attack_bb_from_square(square):
    attack_bb = make_empty_uint64_bitmap()
    hot = np.uint64(1)

    # Diagonal
    if square % 8 == 0:
        attack_bb |= hot << np.uint64(square)
        square += 9
    while not square % 8 == 0:
        attack_bb |= hot << np.uint64(square)
        square += 9
    # Anti-Diagonal
    if square % 8 == 0:
        attack_bb |= hot << np.uint64(square)
        square -= 9
    else:
        while not square % 8 == 0:
            attack_bb |= hot << np.uint64(square)
            square -= 9
        attack_bb |= hot << np.uint64(square)
    return attack_bb


def generate_king_attack_bb_from_square(square):
    attack_bb = make_empty_uint64_bitmap()
    hot = np.uint64(1)
    for i in [0, 8, -8]:
        # North-South
        attack_bb |= hot << np.uint64(square + i)
    for i in [1, 9, -7]:
        # East (mask the A file)
        attack_bb |= hot << np.uint64(square + i) & ~np.uint64(File.A)
    for i in [-1, -9, 7]:
        # West (mask the H file)
        attack_bb |= hot << np.uint64(square + i) & ~np.uint64(File.H)
    return attack_bb


def generate_pawn_attack_bb_from_square(square):
    bitmap = make_empty_uint64_bitmap()
    pass
