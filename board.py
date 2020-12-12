import numpy as np

from bitboard_helpers import make_empty_uint64_bitmap, set_bit, generate_knight_attack_bb_from_square, \
    generate_rank_attack_bb_from_square, generate_file_attack_bb_from_square, generate_diag_attack_bb_from_square, \
    generate_king_attack_bb_from_square, generate_pawn_attack_bb_from_square
from constants import Piece, File, Rank, LIGHT_SQUARES, DARK_SQUARES

BOARD_SIZE = 8
BOARD_SQUARES = BOARD_SIZE ** 2


def _make_knight_attack_bbs():
    knight_attack_map = {}
    for i in range(BOARD_SQUARES):
        knight_attack_map[i] = generate_knight_attack_bb_from_square(i)
    return knight_attack_map


def _make_rank_attack_bbs():
    rank_attack_map = {}
    for i in range(BOARD_SQUARES):
        rank_attack_map[i] = generate_rank_attack_bb_from_square(i)
    return rank_attack_map


def _make_file_attack_bbs():
    file_attack_map = {}
    for i in range(BOARD_SQUARES):
        file_attack_map[i] = generate_file_attack_bb_from_square(i)
    return file_attack_map


def _make_diag_attack_bbs():
    diag_attack_map = {}
    for i in range(BOARD_SQUARES):
        diag_attack_map[i] = generate_diag_attack_bb_from_square(i)
    return diag_attack_map


def _make_king_attack_bbs():
    king_attack_map = {}
    for i in range(BOARD_SQUARES):
        king_attack_map[i] = generate_king_attack_bb_from_square(i)
    return king_attack_map


def _make_pawn_attack_bbs():
    pawn_attack_map = {}
    for i in range(BOARD_SQUARES):
        pawn_attack_map[i] = generate_pawn_attack_bb_from_square(i)
    return pawn_attack_map


class Board:

    def __init__(self):

        # white piece groups
        self.white_R_bb = make_empty_uint64_bitmap()
        self.white_K_bb = make_empty_uint64_bitmap()
        self.white_B_bb = make_empty_uint64_bitmap()
        self.white_P_bb = make_empty_uint64_bitmap()
        self.white_N_bb = make_empty_uint64_bitmap()
        self.white_Q_bb = make_empty_uint64_bitmap()

        # black piece groups
        self.black_R_bb = make_empty_uint64_bitmap()
        self.black_K_bb = make_empty_uint64_bitmap()
        self.black_B_bb = make_empty_uint64_bitmap()
        self.black_P_bb = make_empty_uint64_bitmap()
        self.black_N_bb = make_empty_uint64_bitmap()
        self.black_Q_bb = make_empty_uint64_bitmap()

        self.init_pieces()

        # static bitboards
        self.knight_bbs = _make_knight_attack_bbs()

    # -------------------------------------------------------------
    #  BITBOARD ACCESS: PIECE LOCATIONS
    # -------------------------------------------------------------

    @property
    def white_pieces_bb(self):
        return self.white_P_bb | self.white_R_bb | self.white_N_bb | self.white_B_bb | self.white_K_bb | self.white_Q_bb

    @property
    def black_pieces_bb(self):
        return self.black_P_bb | self.black_R_bb | self.black_N_bb | self.black_B_bb | self.black_K_bb | self.black_Q_bb

    @property
    def empty_squares_bb(self):
        return ~self.occupied_squares_bb

    @property
    def occupied_squares_bb(self):
        return self.white_pieces_bb | self.black_pieces_bb

    # -------------------------------------------------------------
    #  BITBOARD ACCESS: BOARD REGIONS
    # -------------------------------------------------------------

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
    def light_squares_bb(self):
        return np.uint64(LIGHT_SQUARES)

    @property
    def dark_squares_bb(self):
        return np.uint64(DARK_SQUARES)

    # -------------------------------------------------------------
    #  BITBOARD ACCESS: RANKS AND FILES
    # -------------------------------------------------------------

    @property
    def file_a_bb(self):
        return np.uint64(File.hexA)

    @property
    def file_b_bb(self):
        return np.uint64(File.hexB)

    @property
    def file_c_bb(self):
        return np.uint64(File.hexC)

    @property
    def file_d_bb(self):
        return np.uint64(File.hexD)

    @property
    def file_e_bb(self):
        return np.uint64(File.hexE)

    @property
    def file_f_bb(self):
        return np.uint64(File.hexF)

    @property
    def file_g_bb(self):
        return np.uint64(File.hexG)

    @property
    def file_h_bb(self):
        return np.uint64(File.hexH)

    @property
    def rank_1_bb(self):
        return np.uint64(Rank.hex1)

    @property
    def rank_2_bb(self):
        return np.uint64(Rank.hex2)

    @property
    def rank_3_bb(self):
        return np.uint64(Rank.hex3)

    @property
    def rank_4_bb(self):
        return np.uint64(Rank.hex4)

    @property
    def rank_5_bb(self):
        return np.uint64(Rank.hex5)

    @property
    def rank_6_bb(self):
        return np.uint64(Rank.hex6)

    @property
    def rank_7_bb(self):
        return np.uint64(Rank.hex7)

    @property
    def rank_8_bb(self):
        return np.uint64(Rank.hex8)

    # -------------------------------------------------------------
    #  BITBOARD ACCESS: PIECE ATTACKS
    # -------------------------------------------------------------

    @property
    def white_P_east_attacks(self):
        # White pawn east attacks are north east (+9) AND NOT the A File
        return (self.white_P_bb << 9) & (~self.file_a_bb)

    @property
    def white_P_west_attacks(self):
        # White pawn west attacks are north west (+7) AND NOT the H File
        return (self.white_P_bb << 7) & (~self.file_h_bb)

    @property
    def white_pawn_attacks(self):
        return self.white_P_east_attacks | self.white_P_west_attacks

    @property
    def black_pawn_east_attacks(self):
        # Black pawn east attacks are south east (-7) AND NOT the A File
        return (self.white_P_bb >> 7) & (~self.file_a_bb)

    @property
    def black_pawn_west_attacks(self):
        # Black pawn west attacks are south west (-9) AND NOT the H File
        return (self.white_P_bb >> 9) & (~self.file_a_bb)

    @property
    def black_pawn_attacks(self):
        return self.black_pawn_east_attacks | self.black_pawn_west_attacks

    # -------------------------------------------------------------
    #  BOARD SETUP
    # -------------------------------------------------------------

    def init_pieces(self):
        self._set_white()
        self._set_black()

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

    # -------------------------------------------------------------
    #  BOARD UPDATES
    # -------------------------------------------------------------

    def update_position(self, piece_map):
        for key, val in piece_map.items():

            # TODO inefficient

            # White Pieces
            if key == Piece.wP:
                self.white_P_bb = np.uint64(0)
                for bit in val:
                    self.white_P_bb |= set_bit(self.white_P_bb, np.uint64(bit))

            elif key == Piece.wR:
                self.white_R_bb = np.uint64(0)
                for bit in val:
                    self.white_R_bb |= set_bit(self.white_R_bb, np.uint64(bit))

            elif key == Piece.wN:
                self.white_N_bb = np.uint64(0)
                for bit in val:
                    self.white_N_bb |= set_bit(self.white_N_bb, np.uint64(bit))

            elif key == Piece.wB:
                self.white_B_bb = np.uint64(0)
                for bit in val:
                    self.white_B_bb |= set_bit(self.white_B_bb, np.uint64(bit))

            elif key == Piece.wQ:
                self.white_Q_bb = np.uint64(0)
                for bit in val:
                    self.white_Q_bb |= set_bit(self.white_Q_bb, np.uint64(bit))

            elif key == Piece.wK:
                self.white_K_bb = np.uint64(0)
                for bit in val:
                    self.white_K_bb |= set_bit(self.white_K_bb, np.uint64(bit))

            # Black Pieces
            if key == Piece.bP:
                self.black_P_bb = np.uint64(0)
                for bit in val:
                    self.black_P_bb |= set_bit(self.black_P_bb, np.uint64(bit))

            elif key == Piece.bR:
                self.black_R_bb = np.uint64(0)
                for bit in val:
                    self.black_P_bb |= set_bit(self.black_R_bb, np.uint64(bit))

            elif key == Piece.bN:
                self.black_N_bb = np.uint64(0)
                for bit in val:
                    self.black_N_bb |= set_bit(self.black_N_bb, np.uint64(bit))

            elif key == Piece.bB:
                self.black_B_bb = np.uint64(0)
                for bit in val:
                    self.black_B_bb |= set_bit(self.black_B_bb, np.uint64(bit))

            elif key == Piece.bQ:
                self.black_Q_bb = np.uint64(0)
                for bit in val:
                    self.black_Q_bb |= set_bit(self.black_Q_bb, np.uint64(bit))

            elif key == Piece.bK:
                self.black_K_bb = np.uint64(0)
                for bit in val:
                    self.black_K_bb |= set_bit(self.black_K_bb, np.uint64(bit))

    # -------------------------------------------------------------
    #  SLIDING PIECE MOVEMENT
    # -------------------------------------------------------------

    def get_bishop_attack_from(self, square):
        pass

    def get_rook_attack_from(self, square):
        pass

    # -------------------------------------------------------------
    #  PAWN MOVEMENTS
    # -------------------------------------------------------------

    def _make_pawn_attack_bbs(self):
        pass
