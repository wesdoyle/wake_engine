import string

import numpy as np

from constants import Piece, FileSquares as fsq, RankSquares as rsq


class Board:

    def __init__(self, board_size=8):

        self.board_size = board_size

        # white piece groups
        self.white_R_bb = self._make_empty_bitmap()
        self.white_K_bb = self._make_empty_bitmap()
        self.white_B_bb = self._make_empty_bitmap()
        self.white_P_bb = self._make_empty_bitmap()
        self.white_N_bb = self._make_empty_bitmap()
        self.white_Q_bb = self._make_empty_bitmap()

        # black piece groups
        self.black_R_bb = self._make_empty_bitmap()
        self.black_K_bb = self._make_empty_bitmap()
        self.black_B_bb = self._make_empty_bitmap()
        self.black_P_bb = self._make_empty_bitmap()
        self.black_N_bb = self._make_empty_bitmap()
        self.black_Q_bb = self._make_empty_bitmap()

        # current piece attacks bitboard
        self.attack_bb = self._make_empty_bitmap()

        self.init_pieces()

        self.rank_1_bb = self._make_empty_bitmap()
        self.rank_2_bb = self._make_empty_bitmap()
        self.rank_3_bb = self._make_empty_bitmap()
        self.rank_4_bb = self._make_empty_bitmap()
        self.rank_5_bb = self._make_empty_bitmap()
        self.rank_6_bb = self._make_empty_bitmap()
        self.rank_7_bb = self._make_empty_bitmap()
        self.rank_8_bb = self._make_empty_bitmap()
        self.file_a_bb = self._make_empty_bitmap()
        self.file_b_bb = self._make_empty_bitmap()
        self.file_c_bb = self._make_empty_bitmap()
        self.file_d_bb = self._make_empty_bitmap()
        self.file_e_bb = self._make_empty_bitmap()
        self.file_f_bb = self._make_empty_bitmap()
        self.file_g_bb = self._make_empty_bitmap()
        self.file_h_bb = self._make_empty_bitmap()

        self._set_rank_file_bitmaps()

        # static knight attacks
        self.knight_bbs = self._make_knight_attack_bb()

        # static pawn attacks
        self.wP_east_attack_map, self.wP_west_attack_map, self.bP_east_attack_map, self.bP_west_attack_map = \
            self._make_pawn_attack_bbs()

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
    def current_white_pawn_attacks(self):
        pass

    @property
    def current_black_pawn_attacks(self):
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
    def plus1(self, square):
        """East Ray"""
        for i in range(square, self.board_size ** 2, 1):
            self.attack_bb[i] = 1
            if not (i + 1) % 8:
                return

    def plus7(self, square):
        """NorthWest Ray"""
        for i in range(square, self.board_size ** 2, 7):
            self.attack_bb[i] = 1
            if not (i + 1) % 8:
                return

    def plus8(self, square):
        """North Ray"""
        for i in range(square, self.board_size ** 2, 8):
            self.attack_bb[i] = 1
            if not (i + 1) % 8:
                return

    def plus9(self, square):
        """NorthEast Ray"""
        for i in range(square, self.board_size ** 2, 9):
            self.attack_bb[i] = 1
            if not (i + 1) % 8:
                return

    def minus1(self, square):
        """West Ray"""
        for i in range(square, 0, -1):
            self.attack_bb[i] = 1
            if not i % 8:
                return

    def minus7(self, square):
        """SouthEast Ray"""
        for i in range(square, 0, -7):
            self.attack_bb[i] = 1
            if not (i + 1) % 8:
                return

    def minus8(self, square):
        """South Ray"""
        for i in range(square, 0, -8):
            self.attack_bb[i] = 1
            if not (i + 1) % 8:
                return

    def minus9(self, square):
        """SouthWest Ray"""
        for i in range(square, -1, -9):
            self.attack_bb[i] = 1
            if not i % 8:
                return

    def get_bishop_attack_from(self, square):
        pass

    def get_rook_attack_from(self, square):
        pass

    # Pawn Attacks
    def _make_pawn_attack_bbs(self):
        wP_east_attack_map = {}
        wP_west_attack_map = {}
        bP_east_attack_map = {}
        bP_west_attack_map = {}

        for i in range(self.board_size ** 2):
            wP_east_attack_map[i] = self._white_pawn_east_attacks(i)
            wP_west_attack_map[i] = self._white_pawn_west_attacks(i)
            bP_east_attack_map[i] = self._black_pawn_east_attacks(i)
            bP_west_attack_map[i] = self._black_pawn_west_attacks(i)

        return wP_east_attack_map, wP_west_attack_map, bP_east_attack_map, bP_west_attack_map

    @staticmethod
    def _white_pawn_east_attacks(square):
        if square in fsq.h:
            pass
        return np.array(square + 9)

    @staticmethod
    def _white_pawn_west_attacks(square):
        if square in fsq.a:
            pass
        return np.array(square + 7)

    @staticmethod
    def _black_pawn_east_attacks(square):
        if square in fsq.h:
            pass
        return np.array(square - 9)

    @staticmethod
    def _black_pawn_west_attacks(square):
        if square in fsq.a:
            pass
        return np.array(square - 9)

    # Knight Attacks
    def _make_knight_attack_bb(self):
        knight_attack_map = {}
        for i in range(self.board_size ** 2):
            knight_attack_map[i] = self._knight_attacks(i)
        return knight_attack_map

    def _knight_attacks(self, square):
        row_mask = self._make_empty_bitmap()
        col_mask = self._make_empty_bitmap()
        agg_mask = self._make_empty_bitmap()

        # overflow file mask
        if square in fsq.a:
            col_mask = self.file_g_bb | self.file_h_bb
        elif square in fsq.b:
            col_mask = self.file_h_bb
        elif square in fsq.g:
            col_mask = self.file_a_bb
        elif square in fsq.h:
            col_mask = self.file_a_bb | self.file_b_bb

        # overflow ranks mask
        if square in rsq._1:
            row_mask = self.rank_8_bb | self.rank_7_bb
        elif square in rsq._2:
            row_mask = self.rank_8_bb

        # aggregate mask
        if row_mask.any() or col_mask.any():
            agg_mask = row_mask | col_mask

        attacks = self._make_empty_bitmap()

        for i in [0, 6, 15, 17, 10, -6, -15, -17, -10]:
            if square + i >= self.board_size ** 2 or square + i < 0:
                # skip OOB
                continue
            attacks[square + i] = 1
        if agg_mask.any():
            # bit shift the attacks by mask
            attacks = attacks >> agg_mask
        return attacks

    def _make_empty_bitmap(self):
        return np.zeros(self.board_size ** 2, dtype="byte")

    def _set_rank_file_bitmaps(self):
        # todo: faster numpy methods
        for val in fsq.a:
            self.file_a_bb[val] = 1

        for val in fsq.b: self.file_b_bb[val] = 1
        for val in fsq.c: self.file_c_bb[val] = 1
        for val in fsq.d: self.file_d_bb[val] = 1
        for val in fsq.e: self.file_e_bb[val] = 1
        for val in fsq.f: self.file_f_bb[val] = 1
        for val in fsq.g: self.file_g_bb[val] = 1
        for val in fsq.h: self.file_h_bb[val] = 1

        for val in rsq._1: self.rank_1_bb[val] = 1
        for val in rsq._2: self.rank_2_bb[val] = 1
        for val in rsq._3: self.rank_3_bb[val] = 1
        for val in rsq._4: self.rank_4_bb[val] = 1
        for val in rsq._5: self.rank_5_bb[val] = 1
        for val in rsq._6: self.rank_6_bb[val] = 1
        for val in rsq._7: self.rank_7_bb[val] = 1
        for val in rsq._8: self.rank_8_bb[val] = 1

    def _set_white(self):
        for i in range(8, 16):
            self.white_P_bb[i] = 1
        self.white_R_bb[0] = 1
        self.white_R_bb[7] = 1
        self.white_N_bb[1] = 1
        self.white_N_bb[6] = 1
        self.white_B_bb[2] = 1
        self.white_B_bb[5] = 1
        self.white_Q_bb[3] = 1
        self.white_K_bb[4] = 1

    def _set_black(self):
        for i in range(48, 56):
            self.black_P_bb[i] = 1
        self.black_R_bb[63] = 1
        self.black_R_bb[56] = 1
        self.black_N_bb[57] = 1
        self.black_N_bb[62] = 1
        self.black_B_bb[61] = 1
        self.black_B_bb[58] = 1
        self.black_Q_bb[59] = 1
        self.black_K_bb[60] = 1


def pretty_print_bb(bb, board_size=8):
    val = ''
    display_rank = board_size
    board = np.reshape(np.flip(bb), (board_size, board_size))
    for i, row in enumerate(board):
        val += f'{display_rank} '
        display_rank -= 1
        for square in np.flip(row):
            if square:
                val += ' ▓'
                continue
            val += ' ░'
        val += '\n'
    val += '  '
    for char in string.ascii_uppercase[:board_size]:
        val += f' {char}'
    print(val)
