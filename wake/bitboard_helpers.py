import string

import numpy as np

from wake.constants import File, HOT, Square, Rank, DARK_SQUARES, LIGHT_SQUARES, piece_to_glyph

BOARD_SIZE = 8
BOARD_SQUARES = BOARD_SIZE ** 2


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
    return format(bitboard, 'b').zfill(board_squares)


def get_squares_from_bitboard(bitboard: np.uint64) -> list:
    binary_string = get_binary_string(bitboard)
    squares = []
    for i, bit in enumerate(reversed(binary_string)):
        if int(bit):
            squares.append(i)
    return squares


# -------------------------------------------------------------
# BIT QUERYING
# -------------------------------------------------------------

def bitscan_forward(bitboard: np.uint64) -> int:
    """
    Returns the least significant one bit from the provided bitboard
    :param bitboard: bitboard to can
    :return: int significant one bit binary string index
    """
    i = 1
    while not (bitboard >> np.uint64(i)) % 2:
        i += 1
    return i


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


# -------------------------------------------------------------
# DEBUG PRETTY PRINT
# -------------------------------------------------------------

def pprint_bb(bitboard: np.uint64, board_size: int = 8) -> None:
    """
    Pretty-prints the given bitboard as 8 x 8 chess board
    :param bitboard: the bitboard to pretty-print
    :param board_size: the length of the square board
    :return: None
    """
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


def pprint_pieces(piece_map: dict, board_size: int = 8) -> None:
    """
    Prints the given piece map as 8 x 8 chess board using Unicode chess symbols
    :param piece_map: Python dictionary of piece to set of square indices
    :param board_size: the length of the square board
    :return: None
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

def generate_knight_attack_bb_from_square(from_square: int) -> np.uint64:
    """
    Generates a static bitboard of squares attacked by a knight from the provided `square`
    :param from_square: square index of the knight from which to generate attack squares bitboard
    :return: np.uint64 bitboard of attacked squares by a knight on the provided `square`
    """
    attack_bb = make_uint64()
    for i in [6, 15, 17, 10, -6, -15, -17, -10]:
        to_square = from_square + i
        if not 0 <= to_square < 64:
            continue

        attack_bb |= set_bit(attack_bb, to_square)
        # Mask of wrapping
        if from_square in (File.B | File.A):
            attack_bb &= ~(np.uint64(File.hexG | File.hexH))
        if from_square in (File.G | File.H):
            attack_bb &= ~(np.uint64(File.hexA | File.hexB))
    return attack_bb


# -------------------------------------------------------------
#  SLIDING ATTACK PATTERNS
# -------------------------------------------------------------

def get_south_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of south sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the south ray sliding attacks from `square`
    :param from_square: The square from a south-sliding piece attacks
    :return: np.uint64 bitboard of the southern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    for i in range(0, -64, -8):
        to_square = from_square + i
        if not 0 <= to_square < 64:
            continue
        bitboard |= set_bit(bitboard, to_square)
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
    for i in range(0, 64, 8):
        to_square = from_square + i
        if not 0 <= to_square < 64:
            continue
        bitboard |= set_bit(bitboard, to_square)
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
        if from_square % 8 == 0:
            break
        from_square += 1

    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_southeast_ray(bitboard: np.uint64, from_square: int) -> np.uint64:
    """
    Returns a bitboard of northeast sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the northeast ray sliding attacks from `square`
    :param from_square: The square from a northeast-sliding piece attacks
    :return: np.uint64 bitboard of the northeastern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    while 0 <= from_square < 64:
        bitboard |= HOT << np.uint64(from_square)
        if from_square % 8 == 0 or from_square in File.H:
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
        if from_square % 8 == 0 or from_square in File.A:
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
        if from_square % 8 == 0 or from_square in File.H:
            break
        from_square -= 9

    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


def get_northeast_ray(bitboard, from_square):
    """
    Returns a bitboard of northeast sliding piece attacked squares on an otherwise empty board
    :param bitboard: The bitboard representing the northeast ray sliding attacks from `square`
    :param from_square: The square from a northeast-sliding piece attacks
    :return: np.uint64 bitboard of the northeastern squares attacked on an otherwise empty board
    """
    original_from_square = from_square
    while 0 <= from_square < 64:
        bitboard |= HOT << np.uint64(from_square)
        if from_square % 8 == 0 or from_square in File.A:
            break
        from_square += 9

    bitboard = clear_bit(bitboard, original_from_square)
    return bitboard


# -------------------------------------------------------------
#  ATTACK PATTERNS: ROOK
# -------------------------------------------------------------

def generate_rank_attack_bb_from_square(square: int) -> np.uint64:
    """
    Generates rank attacks from the provided square on an otherwise empty bitboard
    :param square: starting square from which to generate rank attacks
    :return: np.uint64 rank attacks bitboard from the provided square
    """
    attack_bb = make_uint64()
    attack_bb = get_north_ray(attack_bb, square)
    attack_bb = get_south_ray(attack_bb, square)
    attack_bb = clear_bit(attack_bb, square)
    return attack_bb


def generate_file_attack_bb_from_square(square: int) -> np.uint64:
    """
    Generates file attacks from the provided square on an otherwise empty bitboard
    :param square: starting square from which to generate file attacks
    :return: np.uint64 file attacks bitboard from the provided square
    """
    attack_bb = make_uint64()
    attack_bb = get_east_ray(attack_bb, square)
    attack_bb = get_west_ray(attack_bb, square)
    attack_bb = clear_bit(attack_bb, square)
    return attack_bb


def generate_rook_attack_bb_from_square(square: int) -> np.uint64:
    """
    Generates rook attacks from the provided square on an otherwise empty bitboard
    :param square: starting square from which to generate rook attacks
    :return: np.uint64 rook attacks bitboard from the provided square
    """
    return generate_file_attack_bb_from_square(square) | generate_rank_attack_bb_from_square(square)


# -------------------------------------------------------------
#  ATTACK PATTERNS: BISHOP
# -------------------------------------------------------------

def generate_diag_attack_bb_from_square(from_square: int) -> np.uint64:
    """
    Generates all diagonal attacks from the provided square on an otherwise empty bitboard
    :param from_square: starting square from which to generate diagonal attacks
    :return: np.uint64 diagonal attacks bitboard from the provided square
    """
    attack_bb = make_uint64()
    original_square = from_square

    attack_bb = get_northeast_ray(attack_bb, from_square)
    attack_bb = get_southwest_ray(attack_bb, from_square)
    attack_bb = get_northwest_ray(attack_bb, from_square)
    attack_bb = get_southeast_ray(attack_bb, from_square)

    attack_bb = clear_bit(attack_bb, original_square)

    return attack_bb


# -------------------------------------------------------------
#  ATTACK PATTERNS: QUEEN
# -------------------------------------------------------------

def generate_queen_attack_bb_from_square(from_square: int) -> np.uint64:
    """
    Returns the queen attack bitboard on an otherwise empty board from the provided square
    :param from_square: starting square from which to generate queen attacks
    :return: np.uint64 bitboard representation of queen attacks on an otherwise empty board
    """
    return generate_diag_attack_bb_from_square(from_square) \
           | generate_file_attack_bb_from_square(from_square) \
           | generate_rank_attack_bb_from_square(from_square)


# -------------------------------------------------------------
#  ATTACK PATTERNS: KING
# -------------------------------------------------------------

def generate_king_attack_bb_from_square(from_square: int) -> np.uint64:
    """
    Generates a static bitboard of squares attacked by a king from the provided `square`
    :param from_square: square index of the king from which to generate attack squares bitboard
    :return: np.uint64 bitboard of attacked squares by a king on the provided `square`
    """
    attack_bb = make_uint64()
    for i in [-1, -7, -8, -9, 1, 7, 8, 9]:
        to_square = from_square + i
        if not 0 <= to_square < 64:
            continue
        attack_bb |= HOT << np.uint64(to_square)
    # Mask of wrapping
    if from_square in File.A:
        attack_bb &= ~np.uint64(File.hexH)
    if from_square in File.H:
        attack_bb &= ~np.uint64(File.hexA)
    return attack_bb


# -------------------------------------------------------------
#  ATTACK PATTERNS: PAWN
# -------------------------------------------------------------

def generate_white_pawn_attack_bb_from_square(from_square: int) -> np.uint64:
    """
    Generates a static bitboard of squares attacked by a white pawn from the provided `square`
    :param from_square: square index of the white pawn from which to generate attack squares bitboard
    :return: np.uint64 bitboard of attacked squares by a white pawn on the provided `square`
    """
    attack_bb = make_uint64()
    for i in [7, 9]:
        to_square = from_square + i
        if not 0 <= to_square < 64:
            continue
        attack_bb |= HOT << np.uint64(to_square)
    # Mask of wrapping
    if from_square in File.A:
        attack_bb &= ~np.uint64(File.hexH)
    if from_square in File.H:
        attack_bb &= ~np.uint64(File.hexA)
    return attack_bb


def generate_black_pawn_attack_bb_from_square(from_square: int) -> np.uint64:
    """
    Generates a static bitboard of squares attacked by a black pawn from the provided `square`
    :param from_square: square index of the black pawn from which to generate attack squares bitboard
    :return: np.uint64 bitboard of attacked squares by a black pawn on the provided `square`
    """
    attack_bb = make_uint64()
    for i in [-7, -9]:
        to_square = from_square + i
        if not 0 <= to_square < 64:
            continue
        attack_bb |= HOT << np.uint64(to_square)
    # Mask of wrapping
    if from_square in File.A:
        attack_bb &= ~np.uint64(File.hexH)
    if from_square in File.H:
        attack_bb &= ~np.uint64(File.hexA)
    return attack_bb


def generate_white_pawn_motion_bb_from_square(from_square: int) -> np.uint64:
    """
    Returns the white pawn motion bitboard on an otherwise empty board from the provided square
    :param from_square: starting square from which to generate white pawn motions
    :return: np.uint64 bitboard representation of white pawn motions on an otherwise empty board
    """
    motion_bb = make_uint64()
    motion_bb |= HOT << np.uint64(from_square + 8)
    if from_square in Rank.x2:
        motion_bb |= HOT << np.uint64(from_square + 16)
    return motion_bb


def generate_black_pawn_motion_bb_from_square(from_square: int) -> np.uint64:
    """
    Returns the black pawn motion bitboard on an otherwise empty board from the provided square
    :param from_square: starting square from which to generate black pawn motions
    :return: np.uint64 bitboard representation of black pawn motions on an otherwise empty board
    """
    motion_bb = make_uint64()
    motion_bb |= HOT << np.uint64(from_square - 8)
    if from_square in Rank.x7:
        motion_bb |= HOT << np.uint64(from_square - 16)
    return motion_bb


# -------------------------------------------------------------
#  ATTACK PATTERN MAPS
# -------------------------------------------------------------

def make_knight_attack_bbs() -> dict:
    """
    Builds a Python dictionary of { square_index: bitboard } to represent static knight attack patterns
    :return: dict {square: knight attack bitboard} static square -> bitboard mapping
    """
    knight_attack_map = {}
    for i in range(BOARD_SQUARES):
        knight_attack_map[i] = generate_knight_attack_bb_from_square(i)
    return knight_attack_map


def make_rank_attack_bbs() -> dict:
    """
    Builds a Python dictionary of { square_index: bitboard } to represent static rank attack patterns
    :return: dict {square: rank attack bitboard} static square -> bitboard mapping
    """
    rank_attack_map = {}
    for i in range(BOARD_SQUARES):
        rank_attack_map[i] = generate_rank_attack_bb_from_square(i)
    return rank_attack_map


def make_file_attack_bbs() -> dict:
    """
    Builds a Python dictionary of { square_index: bitboard } to represent static file attack patterns
    :return: dict {square: file attack bitboard} static square -> bitboard mapping
    """
    file_attack_map = {}
    for i in range(BOARD_SQUARES):
        file_attack_map[i] = generate_file_attack_bb_from_square(i)
    return file_attack_map


def make_diag_attack_bbs() -> dict:
    """
    Builds a Python dictionary of { square_index: bitboard } to represent static diagonal attack patterns
    :return: dict {square: diagonal attack bitboard} static square -> bitboard mapping
    """
    diag_attack_map = {}
    for i in range(BOARD_SQUARES):
        diag_attack_map[i] = generate_diag_attack_bb_from_square(i)
    return diag_attack_map


def make_king_attack_bbs() -> dict:
    """
    Builds a Python dictionary of { square_index: bitboard } to represent static king attack patterns
    :return: dict {square: king attack bitboard} static square -> bitboard mapping
    """
    king_attack_map = {}
    for i in range(BOARD_SQUARES):
        king_attack_map[i] = generate_king_attack_bb_from_square(i)
    return king_attack_map


def make_queen_attack_bbs() -> dict:
    """
    Builds a Python dictionary of { square_index: bitboard } to represent static queen attack patterns
    :return: dict {square: queen attack bitboard} static square -> bitboard mapping
    """
    queen_attack_map = {}
    for i in range(BOARD_SQUARES):
        queen_attack_map[i] = generate_queen_attack_bb_from_square(i)
    return queen_attack_map


def make_rook_attack_bbs() -> dict:
    """
    Builds a Python dictionary of { square_index: bitboard } to represent static rook attack patterns
    :return: dict {square: rook attack bitboard} static square -> bitboard mapping
    """
    rook_attack_map = {}
    for i in range(BOARD_SQUARES):
        rook_attack_map[i] = generate_rook_attack_bb_from_square(i)
    return rook_attack_map


def make_white_pawn_attack_bbs() -> dict:
    """
    Builds a Python dictionary of { square_index: bitboard } to represent static white pawn attack patterns
    :return: dict {square: white pawn attack bitboard} static square -> bitboard mapping
    """
    white_pawn_attack_map = {}
    for i in range(Square.A2, Square.A8):
        white_pawn_attack_map[i] = generate_white_pawn_attack_bb_from_square(i)
    return white_pawn_attack_map


def make_black_pawn_attack_bbs() -> dict:
    """
    Builds a Python dictionary of { square_index: bitboard } to represent static black pawn attack patterns
    :return: dict {square: black pawn attack bitboard} static square -> bitboard mapping
    """
    black_pawn_attack_map = {}
    for i in range(Square.A2, Square.A8):
        black_pawn_attack_map[i] = generate_black_pawn_attack_bb_from_square(i)
    return black_pawn_attack_map


def make_white_pawn_motion_bbs() -> dict:
    """
    Builds a Python dictionary of { square_index: bitboard } to represent static white pawn motion patterns
    :return: dict {square: white pawn motion bitboard} static square -> bitboard mapping
    """
    white_pawn_motion_map = {}
    for i in range(Square.A2, Square.A8):
        white_pawn_motion_map[i] = generate_white_pawn_motion_bb_from_square(i)
    return white_pawn_motion_map


def make_black_pawn_motion_bbs() -> dict:
    """
    Builds a Python dictionary of { square_index: bitboard } to represent static black pawn motion patterns
    :return: dict {square: black pawn motion bitboard} static square -> bitboard mapping
    """
    black_pawn_motion_map = {}
    for i in range(Square.A2, Square.A8):
        black_pawn_motion_map[i] = generate_black_pawn_motion_bb_from_square(i)
    return black_pawn_motion_map


# -------------------------------------------------------------
#  BITBOARD ACCESS: BOARD REGIONS
# -------------------------------------------------------------

def rank_8_bb() -> np.uint64: return np.uint64(Rank.hex8)


def rank_7_bb() -> np.uint64: return np.uint64(Rank.hex7)


def rank_6_bb() -> np.uint64: return np.uint64(Rank.hex6)


def rank_5_bb() -> np.uint64: return np.uint64(Rank.hex5)


def rank_4_bb() -> np.uint64: return np.uint64(Rank.hex4)


def rank_3_bb() -> np.uint64: return np.uint64(Rank.hex3)


def rank_2_bb() -> np.uint64: return np.uint64(Rank.hex2)


def rank_1_bb() -> np.uint64: return np.uint64(Rank.hex1)


def file_h_bb() -> np.uint64: return np.uint64(File.hexH)


def file_g_bb() -> np.uint64: return np.uint64(File.hexG)


def file_f_bb() -> np.uint64: return np.uint64(File.hexF)


def file_e_bb() -> np.uint64: return np.uint64(File.hexE)


def file_d_bb() -> np.uint64: return np.uint64(File.hexD)


def file_c_bb() -> np.uint64: return np.uint64(File.hexC)


def file_b_bb() -> np.uint64: return np.uint64(File.hexB)


def file_a_bb() -> np.uint64: return np.uint64(File.hexA)


def dark_squares_bb() -> np.uint64: return np.uint64(DARK_SQUARES)


def light_squares_bb() -> np.uint64: return np.uint64(LIGHT_SQUARES)


def center_squares_bb():
    return (file_e_bb() | file_d_bb()) & (rank_4_bb() | rank_5_bb())


def flanks_bb(): return file_a_bb() | file_h_bb()


def center_files_bb(): return file_c_bb() | file_d_bb() | file_e_bb() | file_f_bb()


def kingside_bb(): return file_e_bb() | file_f_bb() | file_g_bb() | file_h_bb()


def queenside_bb(): return file_a_bb() | file_b_bb() | file_c_bb() | file_d_bb()
