import numpy as np

LIGHT_SQUARES = 0x55AA55AA55AA55AA
DARK_SQUARES = 0xAA55AA55AA55AA55


class SquarePosition:
    a1 = 0
    b1 = 1
    c1 = 2
    d1 = 3
    e1 = 4
    f1 = 5
    g1 = 6
    h1 = 7

    a2 = 8
    b2 = 9
    c2 = 10
    d2 = 11
    e2 = 12
    f2 = 13
    g2 = 14
    h2 = 15

    a3 = 16
    b3 = 17
    c3 = 18
    d3 = 19
    e3 = 20
    f3 = 21
    g3 = 22
    h3 = 23

    a4 = 24
    b4 = 25
    c4 = 26
    d4 = 27
    e4 = 28
    f4 = 29
    g4 = 30
    h4 = 31

    a5 = 32
    b5 = 33
    c5 = 34
    d5 = 35
    e5 = 36
    f5 = 37
    g5 = 38
    h5 = 39

    a6 = 40
    b6 = 41
    c6 = 42
    d6 = 43
    e6 = 44
    f6 = 45
    g6 = 46
    h6 = 47

    a7 = 48
    b7 = 49
    c7 = 50
    d7 = 51
    e7 = 52
    f7 = 53
    g7 = 54
    h7 = 55

    a8 = 56
    b8 = 57
    c8 = 58
    d8 = 59
    e8 = 60
    f8 = 61
    g8 = 62
    h8 = 63


class File:
    A = {0, 8, 16, 24, 32, 40, 48, 56}
    B = {1, 9, 17, 25, 33, 41, 49, 57}
    C = {2, 10, 18, 26, 34, 42, 50, 58}
    D = {3, 11, 19, 27, 35, 43, 51, 59}
    E = {4, 12, 20, 28, 36, 44, 52, 60}
    F = {5, 13, 21, 29, 37, 45, 53, 61}
    G = {6, 14, 22, 30, 38, 46, 54, 62}
    H = {7, 15, 23, 31, 39, 47, 55, 63}

    hexA = np.array([0x0101010101010101], dtype=np.uint64)
    hexB = np.array([0x0202020202020202], dtype=np.uint64)
    hexC = np.array([0x0404040404040404], dtype=np.uint64)
    hexD = np.array([0x0808080808080808], dtype=np.uint64)
    hexE = np.array([0x1010101010101010], dtype=np.uint64)
    hexF = np.array([0x2020202020202020], dtype=np.uint64)
    hexG = np.array([0x4040404040404040], dtype=np.uint64)
    hexH = np.array([0x8080808080808080], dtype=np.uint64)

    files = [A, B, C, D, E, F, G, H]


class Rank:
    x1 = {0, 1, 2, 3, 4, 5, 6, 7}
    x2 = {8, 9, 10, 11, 12, 13, 14, 15}
    x3 = {16, 17, 18, 19, 20, 21, 22, 23}
    x4 = {24, 25, 26, 27, 28, 29, 30, 31}
    x5 = {32, 33, 34, 35, 36, 37, 38, 39}
    x6 = {40, 41, 42, 43, 44, 45, 46, 47}
    x7 = {48, 49, 50, 51, 52, 53, 54, 55}
    x8 = {56, 57, 58, 59, 60, 61, 62, 63}

    hex1 = np.array([0x00000000000000FF], dtype=np.uint64)
    hex2 = np.array([0x000000000000FF00], dtype=np.uint64)
    hex3 = np.array([0x0000000000FF0000], dtype=np.uint64)
    hex4 = np.array([0x00000000FF000000], dtype=np.uint64)
    hex5 = np.array([0x000000FF00000000], dtype=np.uint64)
    hex6 = np.array([0x0000FF0000000000], dtype=np.uint64)
    hex7 = np.array([0x00FF000000000000], dtype=np.uint64)
    hex8 = np.array([0xFF00000000000000], dtype=np.uint64)

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
