from board import Board as bb
from constants import Color, Piece

class Position:
    """Represents the internal state of a chess position"""

    def __init__(self):
        self.board = Board.get_piece_positions()
        self.to_move = Color.WHITE
        self.castle_rights = { Color.WHITE: True, Color.BLACK: True }
        self.en_passant_target = None # target square
        self.halfmove_clock = 0

        self.piece_map = {}

    def get_piece_locations(self):
        pass

    def set_initial_piece_locs(self):
        for i in range(8, 16):
            self.piece_map[i] = Piece.wP

        piece_map[0] = Piece.wR
        piece_map[7] = Piece.wR
        piece_map[1] = Piece.wN
        piece_map[6] = Piece.wN
        piece_map[2] = Piece.wB
        piece_map[5] = Piece.wB
        piece_map[3] = Piece.wQ
        piece_map[4] = Piece.wK

        for i in range(48, 56):
            self.piece_map[i] = Piece.wP

        piece_map[56] = Piece.bR
        piece_map[63] = Piece.bR
        piece_map[57] = Piece.bN
        piece_map[62] = Piece.bN
        piece_map[58] = Piece.bB
        piece_map[61] = Piece.bB
        piece_map[59] = Piece.bQ
        piece_map[60] = Piece.bK

        for i in range(48, 56):
            self.black_P_bb[i] = 1

    def make_move(self, move):
        if not self.is_legal_move(move):
            print('Illegal move')
            return

        self.update_bitboards(move)
        self.to_move = not self.to_move

    def update_bitboards(self, move):
        pass

