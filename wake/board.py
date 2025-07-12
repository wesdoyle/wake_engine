import numpy as np

from wake.bitboard_helpers import (
    make_uint64,
    set_bit,
    make_knight_attack_bbs,
    make_king_attack_bbs,
    make_white_pawn_attack_bbs,
    make_black_pawn_attack_bbs,
    make_diag_attack_bbs,
    make_rook_attack_bbs,
    make_white_pawn_motion_bbs,
    make_black_pawn_motion_bbs,
    make_queen_attack_bbs,
)
from wake.constants import Piece, Color, File


class Board:

    def __init__(self):

        # white piece groups
        self.white_R_bb = make_uint64()
        self.white_K_bb = make_uint64()
        self.white_B_bb = make_uint64()
        self.white_P_bb = make_uint64()
        self.white_N_bb = make_uint64()
        self.white_Q_bb = make_uint64()

        # black piece groups
        self.black_R_bb = make_uint64()
        self.black_K_bb = make_uint64()
        self.black_B_bb = make_uint64()
        self.black_P_bb = make_uint64()
        self.black_N_bb = make_uint64()
        self.black_Q_bb = make_uint64()

        self.init_pieces()

        # static bitboards
        self.knight_attack_bbs = make_knight_attack_bbs()
        self.bishop_attack_bbs = make_diag_attack_bbs()
        self.king_attack_bbs = make_king_attack_bbs()
        self.rook_attack_bbs = make_rook_attack_bbs()
        self.queen_attack_bbs = make_queen_attack_bbs()

        self.white_pawn_attack_bbs = make_white_pawn_attack_bbs()
        self.black_pawn_attack_bbs = make_black_pawn_attack_bbs()
        self.white_pawn_motion_bbs = make_white_pawn_motion_bbs()
        self.black_pawn_motion_bbs = make_black_pawn_motion_bbs()

    # -------------------------------------------------------------
    #  BITBOARD ACCESS: PIECE LOCATIONS
    # -------------------------------------------------------------

    @property
    def white_pieces_bb(self):
        return (
            self.white_P_bb
            | self.white_R_bb
            | self.white_N_bb
            | self.white_B_bb
            | self.white_K_bb
            | self.white_Q_bb
        )

    @property
    def black_pieces_bb(self):
        return (
            self.black_P_bb
            | self.black_R_bb
            | self.black_N_bb
            | self.black_B_bb
            | self.black_K_bb
            | self.black_Q_bb
        )

    @property
    def empty_squares_bb(self):
        return ~self.occupied_squares_bb

    @property
    def occupied_squares_bb(self):
        return self.white_pieces_bb | self.black_pieces_bb

    @property
    def white_pawn_east_attacks(self):
        # White pawn east attacks are north east (+9) AND NOT the A File
        return (self.white_P_bb << np.uint64(9)) & ~np.uint64(File.hexA)

    @property
    def white_pawn_west_attacks(self):
        # White pawn west attacks are north west (+7) AND NOT the H File
        return (self.white_P_bb << np.uint64(7)) & ~np.uint64(File.hexH)

    @property
    def white_pawn_attacks(self):
        return self.white_pawn_east_attacks | self.white_pawn_west_attacks

    @property
    def black_pawn_east_attacks(self):
        # Black pawn east attacks are south east (-7) AND NOT the A File
        return (self.black_P_bb >> np.uint64(7)) & ~np.uint64(File.hexA)

    @property
    def black_pawn_west_attacks(self):
        # Black pawn west attacks are south west (-9) AND NOT the H File
        return (self.black_P_bb >> np.uint64(9)) & ~np.uint64(File.hexH)

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

    def update_position_bitboards(self, piece_map):
        for key, val in piece_map.items():
            # TODO: make more efficient by storing and updating the piece group that changed

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
                    self.black_R_bb |= set_bit(self.black_R_bb, np.uint64(bit))

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
        return self.bishop_attack_bbs[square]

    def get_rook_attack_from(self, square):
        return self.rook_attack_bbs[square]

    def get_queen_attack_from(self, square):
        return self.queen_attack_bbs[square]

    # -------------------------------------------------------------
    #  PAWN MOVEMENTS
    # -------------------------------------------------------------

    def get_pawn_attack_from(self, color, square):
        if color is Color.WHITE:
            return self.white_pawn_attack_bbs[square]
        return self.black_pawn_attack_bbs[square]

    def get_pawn_movements_from(self, color, square):
        if color is Color.WHITE:
            return self.white_pawn_motion_bbs[square]
        return self.black_pawn_motion_bbs[square]

    # -------------------------------------------------------------
    #  KNIGHT MOVEMENTS
    # -------------------------------------------------------------

    def get_knight_attack_from(self, square):
        return self.knight_attack_bbs[square]

    # -------------------------------------------------------------
    #  KING MOVEMENTS
    # -------------------------------------------------------------

    def get_king_attack_from(self, square):
        return self.king_attack_bbs[square]
