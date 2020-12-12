import string

import numpy as np

from board import BOARD_SQUARES
from constants import File, HOT, Square, Rank, DARK_SQUARES, LIGHT_SQUARES


def make_empty_uint64_bitmap():
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


def generate_queen_attack_bb_from_square(square):
    return generate_diag_attack_bb_from_square(square) \
           | generate_file_attack_bb_from_square(square) \
           | generate_rank_attack_bb_from_square(square)


def generate_rook_attack_bb_from_square(square):
    return generate_file_attack_bb_from_square(square) \
           | generate_rank_attack_bb_from_square(square)


def generate_white_pawn_attack_bb_from_square(square):
    attack_bb = make_empty_uint64_bitmap()
    # Northeast (mask the A file)
    attack_bb |= HOT << np.uint64(square + 9) & ~np.uint64(File.A)
    # Northwest (mask the H file)
    attack_bb |= HOT << np.uint64(square + 7) & ~np.uint64(File.H)
    return attack_bb


def generate_black_pawn_attack_bb_from_square(square):
    attack_bb = make_empty_uint64_bitmap()
    # Southeast (mask the A file)
    attack_bb |= HOT << np.uint64(square - 9) & ~np.uint64(File.A)
    # Southwest (mask the H file)
    attack_bb |= HOT << np.uint64(square - 7) & ~np.uint64(File.H)
    return attack_bb


def generate_white_pawn_motion_bb_from_square(square):
    motion_bb = make_empty_uint64_bitmap()
    motion_bb |= HOT << np.uint64(square + 8)
    if square in Rank.x2:
        motion_bb |= HOT << np.uint64(square + 16)
    return motion_bb


def generate_black_pawn_motion_bb_from_square(square):
    motion_bb = make_empty_uint64_bitmap()
    motion_bb |= HOT << np.uint64(square - 8)
    if square in Rank.x2:
        motion_bb |= HOT << np.uint64(square - 16)
    return motion_bb


# -------------------------------------------------------------
#  ATTACK PATTERN MAPS
# -------------------------------------------------------------

def make_knight_attack_bbs():
    knight_attack_map = {}
    for i in range(BOARD_SQUARES):
        knight_attack_map[i] = generate_knight_attack_bb_from_square(i)
    return knight_attack_map


def make_rank_attack_bbs():
    rank_attack_map = {}
    for i in range(BOARD_SQUARES):
        rank_attack_map[i] = generate_rank_attack_bb_from_square(i)
    return rank_attack_map


def make_file_attack_bbs():
    file_attack_map = {}
    for i in range(BOARD_SQUARES):
        file_attack_map[i] = generate_file_attack_bb_from_square(i)
    return file_attack_map


def make_diag_attack_bbs():
    diag_attack_map = {}
    for i in range(BOARD_SQUARES):
        diag_attack_map[i] = generate_diag_attack_bb_from_square(i)
    return diag_attack_map


def make_king_attack_bbs():
    king_attack_map = {}
    for i in range(BOARD_SQUARES):
        king_attack_map[i] = generate_king_attack_bb_from_square(i)
    return king_attack_map


def make_queen_attack_bbs():
    queen_attack_map = {}
    for i in range(BOARD_SQUARES):
        queen_attack_map[i] = generate_queen_attack_bb_from_square(i)
    return queen_attack_map


def make_rook_attack_bbs():
    rook_attack_map = {}
    for i in range(BOARD_SQUARES):
        rook_attack_map[i] = generate_rook_attack_bb_from_square(i)
    return rook_attack_map


def make_white_pawn_attack_bbs():
    white_pawn_attack_map = {}
    for i in range(Square.A2, Square.A8):
        white_pawn_attack_map[i] = generate_white_pawn_attack_bb_from_square(i)
    return white_pawn_attack_map


def make_black_pawn_attack_bbs():
    black_pawn_attack_map = {}
    for i in range(Square.A2, Square.A8):
        black_pawn_attack_map[i] = generate_black_pawn_attack_bb_from_square(i)
    return black_pawn_attack_map


def make_white_pawn_motion_bbs():
    white_pawn_motion_map = {}
    for i in range(Square.A2, Square.A8):
        white_pawn_motion_map[i] = generate_white_pawn_motion_bb_from_square(i)
    return white_pawn_motion_map


def make_black_pawn_motion_bbs():
    black_pawn_motion_map = {}
    for i in range(Square.A2, Square.A8):
        black_pawn_motion_map[i] = generate_black_pawn_motion_bb_from_square(i)
    return black_pawn_motion_map


# -------------------------------------------------------------
#  BITBOARD ACCESS: BOARD REGIONS
# -------------------------------------------------------------

def rank_8_bb():
    return np.uint64(Rank.hex8)


def rank_7_bb():
    return np.uint64(Rank.hex7)


def rank_6_bb():
    return np.uint64(Rank.hex6)


def rank_5_bb():
    return np.uint64(Rank.hex5)


def rank_4_bb():
    return np.uint64(Rank.hex4)


def rank_3_bb():
    return np.uint64(Rank.hex3)


def rank_2_bb():
    return np.uint64(Rank.hex2)


def rank_1_bb():
    return np.uint64(Rank.hex1)


def file_h_bb():
    return np.uint64(File.hexH)


def file_g_bb():
    return np.uint64(File.hexG)


def file_f_bb():
    return np.uint64(File.hexF)


def file_e_bb():
    return np.uint64(File.hexE)


def file_d_bb():
    return np.uint64(File.hexD)


def file_c_bb():
    return np.uint64(File.hexC)


def file_b_bb():
    return np.uint64(File.hexB)


def file_a_bb():
    return np.uint64(File.hexA)


def dark_squares_bb():
    return np.uint64(DARK_SQUARES)


def light_squares_bb():
    return np.uint64(LIGHT_SQUARES)


def center_squares_bb():
    return (file_e_bb | file_d_bb) & (rank_4_bb | rank_5_bb)


def flanks_bb():
    return file_a_bb | file_h_bb


def center_files_bb():
    return file_c_bb | file_d_bb | file_e_bb | file_f_bb


def kingside_bb():
    return file_e_bb | file_f_bb | file_g_bb | file_h_bb


def queenside_bb():
    return file_a_bb | file_b_bb | file_c_bb | file_d_bb
