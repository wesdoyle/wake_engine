import numpy as np

from constants import FileSquares as fsq, RankSquares as rsq

class Board():

    """
    Scratch notes
        - square-centric representation
        - function to map fen <-> Board state
        - function to map from geometric <-> bitmap
        ---

        - function to bitwise AND all bitmaps

        -> Piece captures (diff color AND),
            Illegal Moves (same color AND), etc.
    """

    def __init__(self):
        self.reset_bb()

        self.rank_1_bb = self.make_empty_bitmap()
        self.rank_2_bb = self.make_empty_bitmap()
        self.rank_3_bb = self.make_empty_bitmap()
        self.rank_4_bb = self.make_empty_bitmap()
        self.rank_5_bb = self.make_empty_bitmap()
        self.rank_6_bb = self.make_empty_bitmap()
        self.rank_7_bb = self.make_empty_bitmap()
        self.rank_8_bb = self.make_empty_bitmap()
        self.file_a_bb = self.make_empty_bitmap()
        self.file_b_bb = self.make_empty_bitmap()
        self.file_c_bb = self.make_empty_bitmap()
        self.file_d_bb = self.make_empty_bitmap()
        self.file_e_bb = self.make_empty_bitmap()
        self.file_f_bb = self.make_empty_bitmap()
        self.file_g_bb = self.make_empty_bitmap()
        self.file_h_bb = self.make_empty_bitmap()

        self.make_rank_file_bitmaps()

        self.white_R_bb = self.make_empty_bitmap()
        self.white_K_bb = self.make_empty_bitmap()
        self.white_B_bb = self.make_empty_bitmap()
        self.white_P_bb = self.make_empty_bitmap()
        self.white_N_bb = self.make_empty_bitmap()
        self.white_Q_bb = self.make_empty_bitmap()

        self.black_R_bb = self.make_empty_bitmap()
        self.black_K_bb = self.make_empty_bitmap()
        self.black_B_bb = self.make_empty_bitmap()
        self.black_P_bb = self.make_empty_bitmap()
        self.black_N_bb = self.make_empty_bitmap()
        self.black_Q_bb = self.make_empty_bitmap()

        self.init_pieces()

        self.occupied_squares_bb = np.vstack((
            self.white_R_bb,
            self.white_N_bb,
            self.white_B_bb,
            self.white_Q_bb,
            self.white_K_bb,
            self.white_P_bb,
            self.black_R_bb,
            self.black_N_bb,
            self.black_B_bb,
            self.black_Q_bb,
            self.black_K_bb,
            self.black_P_bb
        ))

    def init_pieces(self):
        self._set_white()
        self._set_black()

    def reset_bb(self):
        self.bb = self.make_empty_bitmap()

    def plus1(self, square):
        for i in range(square, 64, 1):
            self.bb[i] = 1
            if not (i+1) % 8:
                return

    def plus7(self, square):
        for i in range(square, 64, 7):
            self.bb[i] = 1
            if not (i+1) % 8:
                return

    def plus8(self, square):
        for i in range(square, 64, 8):
            self.bb[i] = 1
            if not (i+1) % 8:
                return

    def plus9(self, square):
        for i in range(square, 64, 9):
            self.bb[i] = 1
            if not (i+1) % 8:
                return

    def minus1(self, square):
        for i in range(square, 0, -1):
            self.bb[i] = 1
            if not (i) % 8:
                return

    def minus7(self, square):
        for i in range(square, 0, -7):
            self.bb[i] = 1
            if not (i+1) % 8:
                return

    def minus8(self, square):
        for i in range(square, 0, -8):
            self.bb[i] = 1
            if not (i+1) % 8:
                return

    def minus9(self, square):
        for i in range(square, -1, -9):
            self.bb[i] = 1
            if not (i) % 8:
                return

    def knight_attacks(self, square):
        mask = self.make_empty_bitmap()
        if square in fsq.a:
            mask = np.bitwise_or(self.file_g_bb, self.file_h_bb)
        elif square in fsq.b:
            mask = self.file_g_bb
        elif square in fsq.g:
            mask = self.file_a_bb
        elif square in fsq.h:
            mask = np.bitwise_or(self.file_a_bb, self.file_b_bb)
        if square in rsq._1:
            mask = np.bitwise_or(self.file_g_bb, self.file_h_bb)
        attacks = self.make_empty_bitmap()
        for i in [0, 6, 15, 17, 10, -6, -15, -17, -10]:
            if square + abs(i) > 64:
                # skip OOB
                continue
            attacks[square + i] = 1
        if 1 in mask:
            # bit shift the attacks by mask
            attacks = attacks >> mask
        return attacks

    def update_occupied_squares_bb(self):
        result = np.zeros(64, "byte")
        for board in self.occupied_squares_bb:
            result = np.bitwise_or(board, result, dtype="byte")
        self.occupied_squares_bb = result

    def get_empty_squares_bb(self):
        return  1 - self.occupied_squares_bb

    def make_empty_bitmap(self):
        return np.zeros(64, dtype="byte")

    def make_rank_file_bitmaps(self):
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

def pretty_print_bb(bb):
    val = ''
    display_rank = 8
    board = np.reshape(np.flip(bb), (8, 8))
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
    for char in 'ABCDEFGH':
        val += f' {char}'
    print(val)
