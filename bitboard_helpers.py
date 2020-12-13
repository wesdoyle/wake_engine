import string

import numpy as np

from constants import File, HOT, Square, Rank, DARK_SQUARES, LIGHT_SQUARES, piece_to_glyph

BOARD_SIZE = 8
BOARD_SQUARES = BOARD_SIZE ** 2


def make_uint64():
    return np.uint64(0)


def get_bitboard_as_bytes(bitboard):
    return bitboard.tobytes()


def get_binary_string(bitboard, board_squares=64):
    return format(bitboard, 'b').zfill(board_squares)


# -------------------------------------------------------------
# BIT QUERYING
# -------------------------------------------------------------

def bitscan_forward(bitboard):
    """ Scans from A1 until we hit a HOT bit """
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
    return np.uint64(bitboard | np.uint64(1) << np.uint64(bit))


def clear_bit(bitboard, bit):
    return bitboard & ~(np.uint64(1) << np.uint64(bit))


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


def pprint_pieces(piece_map, board_size=8):
    """
    Given a piece map, prints the board using Unicode Chess Symbols
    :param piece_map:
    :param board_size:
    :return:
    """
    board = ['░'] * 64
    for piece, squares in piece_map.items():
        for square in squares:
            board[square] = piece_to_glyph[piece]
    board = np.array(board)
    board = np.reshape(board, (8, 8))
    display_rank = board_size
    for i, row in enumerate(reversed(board)):
        res = f'{display_rank} '
        display_rank -= 1
        for glyph in row:
            res += f' {glyph}'
        print(res)
    res = '  '
    for char in string.ascii_uppercase[:board_size]:
        res += f' {char}'
    print(res)


# -------------------------------------------------------------
#  ATTACK PATTERNS: KNIGHT
# -------------------------------------------------------------

def generate_knight_attack_bb_from_square(square):
    attack_bb = make_uint64()
    for i in [6, 15, 17, 10, -6, -15, -17, -10]:
        attack_bb |= set_bit(attack_bb, square + i)
        # Mask of wrapping
        if square in (File.B | File.A):
            attack_bb &= ~(np.uint64(File.hexG | File.hexH))
        if square in (File.G | File.H):
            attack_bb &= ~(np.uint64(File.hexA | File.hexB))
    return attack_bb


# -------------------------------------------------------------
#  ATTACK PATTERNS: ROOK
# -------------------------------------------------------------

def get_south_ray(bitboard, square):
    for i in range(0, -64, -8):
        bitboard |= set_bit(bitboard, square + i)
    bitboard = clear_bit(bitboard, square)
    return bitboard


def get_north_ray(bitboard, square):
    for i in range(0, 64, 8):
        bitboard |= set_bit(bitboard, square + i)
    return bitboard


def get_west_ray(bitboard, square):
    if square % 8 == 0:
        bitboard |= HOT << np.uint64(square)
        square -= 1
    else:
        while not square % 8 == 0:
            bitboard |= HOT << np.uint64(square)
            square -= 1
        bitboard |= HOT << np.uint64(square)
    return bitboard


def get_east_ray(bitboard, square):
    if not square % 8:
        bitboard |= HOT << np.uint64(square)
        square += 1
    while not square % 8 == 0:
        bitboard |= HOT << np.uint64(square)
        square += 1
    return bitboard


def generate_rank_attack_bb_from_square(square):
    attack_bb = make_uint64()
    attack_bb = get_north_ray(attack_bb, square)
    attack_bb = get_south_ray(attack_bb, square)
    attack_bb = clear_bit(attack_bb, square)
    return attack_bb


def generate_file_attack_bb_from_square(square):
    attack_bb = make_uint64()
    attack_bb = get_east_ray(attack_bb, square)
    attack_bb = get_west_ray(attack_bb, square)
    attack_bb = clear_bit(attack_bb, square)
    return attack_bb


def generate_rook_attack_bb_from_square(square):
    return generate_file_attack_bb_from_square(square) \
           | generate_rank_attack_bb_from_square(square)


# -------------------------------------------------------------
#  ATTACK PATTERNS: BISHOP
# -------------------------------------------------------------

def generate_diag_attack_bb_from_square(square):
    attack_bb = make_uint64()
    original_square = square

    attack_bb = get_northeast_ray(attack_bb, square)
    attack_bb = get_southwest_ray(attack_bb, square)
    attack_bb = get_northwest_ray(attack_bb, square)
    attack_bb = get_southeast_ray(attack_bb, square)

    attack_bb = clear_bit(attack_bb, original_square)

    return attack_bb


def get_southeast_ray(bitboard, from_square):
    if from_square % 8 == 0 and from_square not in File.H:
        bitboard |= HOT << np.uint64(from_square)
        from_square -= 7
    while not from_square % 8 == 0 and from_square not in File.H:
        bitboard |= HOT << np.uint64(from_square)
        from_square -= 7
    bitboard |= HOT << np.uint64(from_square)
    return bitboard


def get_northwest_ray(bitboard, square):
    if square % 8 == 0 and square not in File.A:
        bitboard |= HOT << np.uint64(square)
        square += 7
    while not square % 8 == 0 and square not in File.A:
        bitboard |= HOT << np.uint64(square)
        square += 7
    bitboard |= HOT << np.uint64(square)
    return bitboard


def get_southwest_ray(bitboard, square):
    if square % 8 == 0:
        bitboard |= HOT << np.uint64(square)
        square -= 9
    else:
        while not square % 8 == 0:
            bitboard |= HOT << np.uint64(square)
            square -= 9
        bitboard |= HOT << np.uint64(square)
    return bitboard


def get_northeast_ray(bitboard, square):
    if square % 8 == 0:
        bitboard |= HOT << np.uint64(square)
        square += 9
    while not square % 8 == 0:
        bitboard |= HOT << np.uint64(square)
        square += 9
    return bitboard


# -------------------------------------------------------------
#  ATTACK PATTERNS: QUEEN
# -------------------------------------------------------------

def generate_queen_attack_bb_from_square(square):
    return generate_diag_attack_bb_from_square(square) \
           | generate_file_attack_bb_from_square(square) \
           | generate_rank_attack_bb_from_square(square)


# -------------------------------------------------------------
#  ATTACK PATTERNS: KING
# -------------------------------------------------------------

def generate_king_attack_bb_from_square(square):
    attack_bb = make_uint64()
    for i in [8, -8]:
        # North-South
        attack_bb |= HOT << np.uint64(square + i)
    for i in [1, 9, -7]:
        # East (mask the A file)
        attack_bb |= HOT << np.uint64(square + i) & ~np.uint64(File.hexA)
    for i in [-1, -9, 7]:
        # West (mask the H file)
        attack_bb |= HOT << np.uint64(square + i) & ~np.uint64(File.hexH)
    return attack_bb


# -------------------------------------------------------------
#  ATTACK PATTERNS: PAWN
# -------------------------------------------------------------

def generate_white_pawn_attack_bb_from_square(square):
    attack_bb = make_uint64()
    # Northeast (mask the A file)
    attack_bb |= HOT << np.uint64(square + 9) & ~np.uint64(File.hexA)
    # Northwest (mask the H file)
    attack_bb |= HOT << np.uint64(square + 7) & ~np.uint64(File.hexH)
    return attack_bb


def generate_black_pawn_attack_bb_from_square(square):
    attack_bb = make_uint64()
    # Southeast (mask the A file)
    attack_bb |= HOT << np.uint64(square - 9) & ~np.uint64(File.hexA)
    # Southwest (mask the H file)
    attack_bb |= HOT << np.uint64(square - 7) & ~np.uint64(File.hexH)
    return attack_bb


def generate_white_pawn_motion_bb_from_square(square):
    motion_bb = make_uint64()
    motion_bb |= HOT << np.uint64(square + 8)
    if square in Rank.x2:
        motion_bb |= HOT << np.uint64(square + 16)
    return motion_bb


def generate_black_pawn_motion_bb_from_square(square):
    motion_bb = make_uint64()
    motion_bb |= HOT << np.uint64(square - 8)
    if square in Rank.x7:
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

def rank_8_bb(): return np.uint64(Rank.hex8)


def rank_7_bb(): return np.uint64(Rank.hex7)


def rank_6_bb(): return np.uint64(Rank.hex6)


def rank_5_bb(): return np.uint64(Rank.hex5)


def rank_4_bb(): return np.uint64(Rank.hex4)


def rank_3_bb(): return np.uint64(Rank.hex3)


def rank_2_bb(): return np.uint64(Rank.hex2)


def rank_1_bb(): return np.uint64(Rank.hex1)


def file_h_bb(): return np.uint64(File.hexH)


def file_g_bb(): return np.uint64(File.hexG)


def file_f_bb(): return np.uint64(File.hexF)


def file_e_bb(): return np.uint64(File.hexE)


def file_d_bb(): return np.uint64(File.hexD)


def file_c_bb(): return np.uint64(File.hexC)


def file_b_bb(): return np.uint64(File.hexB)


def file_a_bb(): return np.uint64(File.hexA)


def dark_squares_bb(): return np.uint64(DARK_SQUARES)


def light_squares_bb(): return np.uint64(LIGHT_SQUARES)


def center_squares_bb(): return (file_e_bb | file_d_bb) & (rank_4_bb | rank_5_bb)


def flanks_bb(): return file_a_bb | file_h_bb


def center_files_bb(): return file_c_bb | file_d_bb | file_e_bb | file_f_bb


def kingside_bb(): return file_e_bb | file_f_bb | file_g_bb | file_h_bb


def queenside_bb(): return file_a_bb | file_b_bb | file_c_bb | file_d_bb
