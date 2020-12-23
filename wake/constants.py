import numpy as np

LIGHT_SQUARES = np.uint64(0x55AA55AA55AA55AA)
DARK_SQUARES = ~LIGHT_SQUARES

HOT = np.uint64(1)


class Square:
    A1 = np.uint64(0)
    B1 = np.uint64(1)
    C1 = np.uint64(2)
    D1 = np.uint64(3)
    E1 = np.uint64(4)
    F1 = np.uint64(5)
    G1 = np.uint64(6)
    H1 = np.uint64(7)

    A2 = np.uint64(8)
    B2 = np.uint64(9)
    C2 = np.uint64(10)
    D2 = np.uint64(11)
    E2 = np.uint64(12)
    F2 = np.uint64(13)
    G2 = np.uint64(14)
    H2 = np.uint64(15)

    A3 = np.uint64(16)
    B3 = np.uint64(17)
    C3 = np.uint64(18)
    D3 = np.uint64(19)
    E3 = np.uint64(20)
    F3 = np.uint64(21)
    G3 = np.uint64(22)
    H3 = np.uint64(23)

    A4 = np.uint64(24)
    B4 = np.uint64(25)
    C4 = np.uint64(26)
    D4 = np.uint64(27)
    E4 = np.uint64(28)
    F4 = np.uint64(29)
    G4 = np.uint64(30)
    H4 = np.uint64(31)

    A5 = np.uint64(32)
    B5 = np.uint64(33)
    C5 = np.uint64(34)
    D5 = np.uint64(35)
    E5 = np.uint64(36)
    F5 = np.uint64(37)
    G5 = np.uint64(38)
    H5 = np.uint64(39)

    A6 = np.uint64(40)
    B6 = np.uint64(41)
    C6 = np.uint64(42)
    D6 = np.uint64(43)
    E6 = np.uint64(44)
    F6 = np.uint64(45)
    G6 = np.uint64(46)
    H6 = np.uint64(47)

    A7 = np.uint64(48)
    B7 = np.uint64(49)
    C7 = np.uint64(50)
    D7 = np.uint64(51)
    E7 = np.uint64(52)
    F7 = np.uint64(53)
    G7 = np.uint64(54)
    H7 = np.uint64(55)

    A8 = np.uint64(56)
    B8 = np.uint64(57)
    C8 = np.uint64(58)
    D8 = np.uint64(59)
    E8 = np.uint64(60)
    F8 = np.uint64(61)
    G8 = np.uint64(62)
    H8 = np.uint64(63)


square_map = {
    0: 'a1',
    1: 'b1',
    2: 'c1',
    3: 'd1',
    4: 'e1',
    5: 'f1',
    6: 'g1',
    7: 'h1',

    8: 'a2',
    9: 'b2',
    10: 'c2',
    11: 'd2',
    12: 'e2',
    13: 'f2',
    14: 'g2',
    15: 'h2',

    16: 'a3',
    17: 'b3',
    18: 'c3',
    19: 'd3',
    20: 'e3',
    21: 'f3',
    22: 'g3',
    23: 'h3',

    24: 'a4',
    25: 'b4',
    26: 'c4',
    27: 'd4',
    28: 'e4',
    29: 'f4',
    30: 'g4',
    31: 'h4',

    32: 'a5',
    33: 'b5',
    34: 'c5',
    35: 'd5',
    36: 'e5',
    37: 'f5',
    38: 'g5',
    39: 'h5',

    40: 'a6',
    41: 'b6',
    42: 'c6',
    43: 'd6',
    44: 'e6',
    45: 'f6',
    46: 'g6',
    47: 'h6',

    48: 'a7',
    49: 'b7',
    50: 'c7',
    51: 'd7',
    52: 'e7',
    53: 'f7',
    54: 'g7',
    55: 'h7',

    56: 'a8',
    57: 'b8',
    58: 'c8',
    59: 'd8',
    60: 'e8',
    61: 'f8',
    62: 'g8',
    63: 'h8',

}


class File:
    A = {0, 8, 16, 24, 32, 40, 48, 56}
    B = {1, 9, 17, 25, 33, 41, 49, 57}
    C = {2, 10, 18, 26, 34, 42, 50, 58}
    D = {3, 11, 19, 27, 35, 43, 51, 59}
    E = {4, 12, 20, 28, 36, 44, 52, 60}
    F = {5, 13, 21, 29, 37, 45, 53, 61}
    G = {6, 14, 22, 30, 38, 46, 54, 62}
    H = {7, 15, 23, 31, 39, 47, 55, 63}

    hexA = np.uint64(0x0101010101010101)
    hexB = np.uint64(0x0202020202020202)
    hexC = np.uint64(0x0404040404040404)
    hexD = np.uint64(0x0808080808080808)
    hexE = np.uint64(0x1010101010101010)
    hexF = np.uint64(0x2020202020202020)
    hexG = np.uint64(0x4040404040404040)
    hexH = np.uint64(0x8080808080808080)

    files = [A, B, C, D, E, F, G, H]


class CastleRoute:
    WhiteKingside = np.uint64(0x70)  # E1,F1,G1
    WhiteQueenside = np.uint64(0x1c)  # E1, D1, C1
    BlackKingside = np.uint64(0x1c00000000000000)  # E8,F8,G8
    BlackQueenside = np.uint64(0x7000000000000000)  # E8, D8, C8


class Rank:
    x1 = {0, 1, 2, 3, 4, 5, 6, 7}
    x2 = {8, 9, 10, 11, 12, 13, 14, 15}
    x3 = {16, 17, 18, 19, 20, 21, 22, 23}
    x4 = {24, 25, 26, 27, 28, 29, 30, 31}
    x5 = {32, 33, 34, 35, 36, 37, 38, 39}
    x6 = {40, 41, 42, 43, 44, 45, 46, 47}
    x7 = {48, 49, 50, 51, 52, 53, 54, 55}
    x8 = {56, 57, 58, 59, 60, 61, 62, 63}

    hex1 = np.uint64(0x00000000000000FF)
    hex2 = np.uint64(0x000000000000FF00)
    hex3 = np.uint64(0x0000000000FF0000)
    hex4 = np.uint64(0x00000000FF000000)
    hex5 = np.uint64(0x000000FF00000000)
    hex6 = np.uint64(0x0000FF0000000000)
    hex7 = np.uint64(0x00FF000000000000)
    hex8 = np.uint64(0xFF00000000000000)

    ranks = [x1, x2, x3, x4, x5, x6, x7, x8]


class Color:
    WHITE = 0
    BLACK = 1



class Piece:
    EMPTY = 0

    wP = 1
    wR = 2
    wN = 3
    wB = 4
    wQ = 5
    wK = 6

    bP = 7
    bR = 8
    bN = 9
    bB = 10
    bQ = 11
    bK = 12

    white_pieces = {wP, wR, wN, wB, wQ, wK}
    black_pieces = {bP, bR, bN, bB, bQ, bK}

    PAWN = "pawn"
    ROOK = "rook"
    KNIGHT = "knight"
    BISHOP = "bishop"
    QUEEN = "queen"
    KING = "king"


piece_to_glyph = {
    Piece.wP: "♙",
    Piece.bP: "♟︎",
    Piece.wR: "♖",
    Piece.bR: "♜",
    Piece.wN: "♘",
    Piece.bN: "♞",
    Piece.wB: "♗",
    Piece.bB: "♝",
    Piece.wQ: "♕",
    Piece.bQ: "♛",
    Piece.wK: "♔",
    Piece.bK: "♚",
}

piece_to_value = {
    Piece.wP: 1,
    Piece.bP: 1,
    Piece.wR: 5,
    Piece.bR: 5,
    Piece.wN: 3,
    Piece.bN: 3,
    Piece.wB: 3,
    Piece.bB: 3,
    Piece.wQ: 9,
    Piece.bQ: 9,
    Piece.wK: np.inf,
    Piece.bK: np.inf,
}

user_promotion_input = {
    "queen": Piece.QUEEN,
    "q": Piece.QUEEN,
    "rook": Piece.ROOK,
    "r": Piece.ROOK,
    "knight": Piece.KNIGHT,
    "k": Piece.KNIGHT,
    "n": Piece.KNIGHT,
    "bishop": Piece.BISHOP,
    "b": Piece.BISHOP,
}

white_promotion_map = {
    Piece.QUEEN: Piece.wQ,
    Piece.ROOK: Piece.wR,
    Piece.KNIGHT: Piece.wN,
    Piece.BISHOP: Piece.wB,
}

black_promotion_map = {
    Piece.QUEEN: Piece.bQ,
    Piece.ROOK: Piece.bR,
    Piece.KNIGHT: Piece.bN,
    Piece.BISHOP: Piece.bB,
}

UCI_INPUT_REGEX = r'(?i)^([a-h]{1}[1-8]{1}){2}(?<=[1-9])$|^([a-h]{1}[1-8]{1}){2}(?<=[8])([qnrb])$'

