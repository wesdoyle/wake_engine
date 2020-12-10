import string

import numpy as np

from constants import Piece, File, Rank

BOARD_SIZE = 8
BOARD_SQUARES = BOARD_SIZE ** 2


def make_empty_uint64_bitmap():
    """
    Returns a numpy array of one uint64 zero value
    :return:
    """
    return np.zeros(1, dtype=np.uint64)


def get_bitboard_as_bytes(bitboard):
    return bitboard.tobytes()


def get_binary_string(bitboard):
    return format(bitboard, 'b').zfill(BOARD_SQUARES)


def set_bit(bitboard, bit):
    return bitboard | (1 << bit)


def clear_bit(bitboard, bit):
    return bitboard & ~(1 << bit)


def pprint_bb(bitboard):
    bitboard = get_binary_string(bitboard)
    val = ''
    display_rank = BOARD_SIZE
    board = [bitboard[i:i + 8] for i in range(0, len(bitboard), BOARD_SIZE)]
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
    for char in string.ascii_uppercase[:BOARD_SIZE]:
        val += f' {char}'
    print(val)


class Board:

    def __init__(self):

        # white piece groups
        self.white_R_bb = make_empty_uint64_bitmap()[0]
        self.white_K_bb = make_empty_uint64_bitmap()[0]
        self.white_B_bb = make_empty_uint64_bitmap()[0]
        self.white_P_bb = make_empty_uint64_bitmap()[0]
        self.white_N_bb = make_empty_uint64_bitmap()[0]
        self.white_Q_bb = make_empty_uint64_bitmap()[0]

        # black piece groups
        self.black_R_bb = make_empty_uint64_bitmap()[0]
        self.black_K_bb = make_empty_uint64_bitmap()[0]
        self.black_B_bb = make_empty_uint64_bitmap()[0]
        self.black_P_bb = make_empty_uint64_bitmap()[0]
        self.black_N_bb = make_empty_uint64_bitmap()[0]
        self.black_Q_bb = make_empty_uint64_bitmap()[0]

        self.init_pieces()

        self.rank_1_bb = make_empty_uint64_bitmap()[0]
        self.rank_2_bb = make_empty_uint64_bitmap()[0]
        self.rank_3_bb = make_empty_uint64_bitmap()[0]
        self.rank_4_bb = make_empty_uint64_bitmap()[0]
        self.rank_5_bb = make_empty_uint64_bitmap()[0]
        self.rank_6_bb = make_empty_uint64_bitmap()[0]
        self.rank_7_bb = make_empty_uint64_bitmap()[0]
        self.rank_8_bb = make_empty_uint64_bitmap()[0]
        self.file_a_bb = make_empty_uint64_bitmap()[0]
        self.file_b_bb = make_empty_uint64_bitmap()[0]
        self.file_c_bb = make_empty_uint64_bitmap()[0]
        self.file_d_bb = make_empty_uint64_bitmap()[0]
        self.file_e_bb = make_empty_uint64_bitmap()[0]
        self.file_f_bb = make_empty_uint64_bitmap()[0]
        self.file_g_bb = make_empty_uint64_bitmap()[0]
        self.file_h_bb = make_empty_uint64_bitmap()[0]

        self._set_rank_file_bitmaps()

        # static knight attacks
        self.knight_bbs = self._make_knight_attack_bb()


    @property
    def white_pieces_bb(self):
        return self.white_P_bb | self.white_R_bb | self.white_N_bb | self.white_B_bb | self.white_K_bb | self.white_Q_bb

    @property
    def black_pieces_bb(self):
        return self.black_P_bb | self.black_R_bb | self.black_N_bb | self.black_B_bb | self.black_K_bb | self.black_Q_bb

    @property
    def empty_squares_bb(self):
        return 1 - self.occupied_squares_bb

    @property
    def occupied_squares_bb(self):
        return self.white_pieces_bb | self.black_pieces_bb

    @property
    def queenside_bb(self):
        return self.file_a_bb | self.file_b_bb | self.file_c_bb | self.file_d_bb

    @property
    def kingside_bb(self):
        return self.file_e_bb | self.file_f_bb | self.file_g_bb | self.file_h_bb

    @property
    def center_files_bb(self):
        return self.file_c_bb | self.file_d_bb | self.file_e_bb | self.file_f_bb

    @property
    def flanks_bb(self):
        return self.file_a_bb | self.file_h_bb

    @property
    def center_squares_bb(self):
        return (self.file_e_bb | self.file_d_bb) & (self.rank_4_bb | self.rank_5_bb)

    @property
    def white_P_east_attacks(self):
        # White pawn east attacks are north east (9) AND NOT the A File
        return (self.white_P_bb << 9) & (~self.file_a_bb)

    @property
    def white_P_west_attacks(self):
        # White pawn east attacks are north west (7) AND NOT the H File
        return (self.white_P_bb << 7) & (~self.file_h_bb)

    @property
    def white_pawn_attacks(self):
        return self.white_P_east_attacks | self.white_P_west_attacks

    @property
    def black_pawn_east_attacks(self):
        pass

    @property
    def black_pawn_west_attacks(self):
        pass

    @property
    def black_pawn_attacks(self):
        pass

    def init_pieces(self):
        self._set_white()
        self._set_black()

    def update_position(self, piece_map):
        for key, val in piece_map.items():
            # white pieces
            if key == Piece.wP:
                self.white_P_bb.fill(0)
                np.put(self.white_P_bb, list(val), 1)
            elif key == Piece.wR:
                self.white_R_bb.fill(0)
                np.put(self.white_R_bb, list(val), 1)
            elif key == Piece.wN:
                self.white_N_bb.fill(0)
                np.put(self.white_N_bb, list(val), 1)
            elif key == Piece.wB:
                self.white_B_bb.fill(0)
                np.put(self.white_B_bb, list(val), 1)
            elif key == Piece.wQ:
                self.white_Q_bb.fill(0)
                np.put(self.white_Q_bb, list(val), 1)
            elif key == Piece.wK:
                self.white_K_bb.fill(0)
                np.put(self.white_K_bb, list(val), 1)
            # white pieces
            if key == Piece.bP:
                self.black_P_bb.fill(0)
                np.put(self.black_P_bb, list(val), 1)
            elif key == Piece.bR:
                self.black_R_bb.fill(0)
                np.put(self.black_R_bb, list(val), 1)
            elif key == Piece.bN:
                self.black_N_bb.fill(0)
                np.put(self.black_N_bb, list(val), 1)
            elif key == Piece.bB:
                self.black_B_bb.fill(0)
                np.put(self.black_B_bb, list(val), 1)
            elif key == Piece.bQ:
                self.black_Q_bb.fill(0)
                np.put(self.black_Q_bb, list(val), 1)
            elif key == Piece.bK:
                self.black_K_bb.fill(0)
                np.put(self.black_K_bb, list(val), 1)

    # Sliding piece movement
    def make_east_ray(self, square):
        for i in range(square, BOARD_SQUARES, 1):
            self.attack_bb[i] = 1
            if not (i + 1) % 8:
                return

    def make_north_west_ray(self, square):
        for i in range(square, BOARD_SQUARES, BOARD_SIZE - 1):
            self.attack_bb[i] = 1
            if not (i + 1) % 8:
                return

    def make_north_ray(self, square):
        for i in range(square, BOARD_SQUARES, BOARD_SIZE):
            self.attack_bb[i] = 1
            if not (i + 1) % BOARD_SIZE:
                return

    def make_north_east_ray(self, square):
        for i in range(square, BOARD_SQUARES, BOARD_SIZE + 1):
            self.attack_bb[i] = 1
            if not (i + 1) % BOARD_SIZE:
                return

    def make_west_ray(self, square):
        for i in range(square, 0, -1):
            self.attack_bb[i] = 1
            if not i % BOARD_SIZE:
                return

    def make_south_east_ray(self, square):
        for i in range(square, 0, -(BOARD_SIZE - 1)):
            self.attack_bb[i] = 1
            if not (i + 1) % BOARD_SIZE:
                return

    def make_south_ray(self, square):
        for i in range(square, 0, -BOARD_SIZE):
            self.attack_bb[i] = 1
            if not (i + 1) % BOARD_SIZE:
                return

    def make_south_west_ray(self, square):
        for i in range(square, -1, -BOARD_SIZE - 1):
            self.attack_bb[i] = 1
            if not i % BOARD_SIZE:
                return

    def get_bishop_attack_from(self, square):
        pass

    def get_rook_attack_from(self, square):
        pass

    # Pawn Attacks
    def _make_pawn_attack_bbs(self):
        pass

    # Knight Attacks
    def _make_knight_attack_bb(self):
        knight_attack_map = {}
        for i in range(BOARD_SQUARES):
            knight_attack_map[i] = self._knight_attacks(i)
        return knight_attack_map

    def _knight_attacks(self, square):
        row_mask = make_empty_uint64_bitmap()
        col_mask = make_empty_uint64_bitmap()
        agg_mask = make_empty_uint64_bitmap()

        # overflow file mask
        if square in File.A:
            col_mask = self.file_g_bb | self.file_h_bb
        elif square in File.B:
            col_mask = self.file_h_bb
        elif square in File.G:
            col_mask = self.file_a_bb
        elif square in File.H:
            col_mask = self.file_a_bb | self.file_b_bb

        # overflow ranks mask
        if square in Rank.x1:
            row_mask = self.rank_8_bb | self.rank_7_bb
        elif square in Rank.x2:
            row_mask = self.rank_8_bb

        # aggregate mask
        if row_mask.any() or col_mask.any():
            agg_mask = row_mask | col_mask

        attacks = make_empty_uint64_bitmap()

        for i in [0, 6, 15, 17, 10, -6, -15, -17, -10]:
            if square + i >= BOARD_SQUARES or square + i < 0:
                # skip OOB
                continue
            attacks[square + i] = 1
        if agg_mask.any():
            # bit shift the attacks by mask
            attacks = attacks >> agg_mask
        return attacks

    def _set_rank_file_bitmaps(self):
        for sq in File.A:
            self.file_a_bb |= set_bit(self.file_a_bb, sq)
        for sq in File.B:
            self.file_b_bb |= set_bit(self.file_b_bb, sq)
        for sq in File.C:
            self.file_c_bb |= set_bit(self.file_b_bb, sq)
        for sq in File.D:
            self.file_d_bb |= set_bit(self.file_b_bb, sq)
        for sq in File.E:
            self.file_e_bb |= set_bit(self.file_b_bb, sq)
        for sq in File.F:
            self.file_f_bb |= set_bit(self.file_b_bb, sq)
        for sq in File.G:
            self.file_g_bb |= set_bit(self.file_b_bb, sq)
        for sq in File.H:
            self.file_h_bb |= set_bit(self.file_b_bb, sq)

        for sq in Rank.x1:
            self.rank_1_bb |= set_bit(self.rank_1_bb, sq)
        for sq in Rank.x2:
            self.rank_2_bb |= set_bit(self.rank_2_bb, sq)
        for sq in Rank.x3:
            self.rank_3_bb |= set_bit(self.rank_3_bb, sq)
        for sq in Rank.x4:
            self.rank_4_bb |= set_bit(self.rank_4_bb, sq)
        for sq in Rank.x5:
            self.rank_5_bb |= set_bit(self.rank_5_bb, sq)
        for sq in Rank.x6:
            self.rank_6_bb |= set_bit(self.rank_6_bb, sq)
        for sq in Rank.x7:
            self.rank_7_bb |= set_bit(self.rank_7_bb, sq)
        for sq in Rank.x8:
            self.rank_8_bb |= set_bit(self.rank_8_bb, sq)

    def _set_white(self):
        for i in range(8, 16):
            self.white_P_bb |= set_bit(self.white_P_bb, i)
        self.white_R_bb |= set_bit(self.white_R_bb, 0)
        self.white_R_bb |= set_bit(self.white_R_bb, 7)
        self.white_N_bb |= set_bit(self.white_N_bb, 1)
        self.white_N_bb |= set_bit(self.white_N_bb, 6)
        self.white_B_bb |= set_bit(self.white_B_bb, 2)
        self.white_B_bb |= set_bit(self.white_B_bb, 5)
        self.white_Q_bb |= set_bit(self.white_Q_bb, 3)
        self.white_K_bb |= set_bit(self.white_K_bb, 4)

    def _set_black(self):
        for bit in range(48, 56):
            self.black_P_bb |= set_bit(self.black_P_bb, bit)
        self.black_R_bb |= set_bit(self.black_R_bb, 63)
        self.black_R_bb |= set_bit(self.black_R_bb, 56)
        self.black_N_bb |= set_bit(self.black_N_bb, 57)
        self.black_N_bb |= set_bit(self.black_N_bb, 62)
        self.black_B_bb |= set_bit(self.black_B_bb, 61)
        self.black_B_bb |= set_bit(self.black_B_bb, 58)
        self.black_Q_bb |= set_bit(self.black_Q_bb, 59)
        self.black_K_bb |= set_bit(self.black_K_bb, 60)
