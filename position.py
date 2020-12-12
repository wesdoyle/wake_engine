from board import Board
from constants import Color, Piece


def generate_fen():
    """ TODO: Generate FEN for the current state """
    return '-- TODO: generate FEN --'


def is_legal_move(move):
    """ TODO: quasi-legal move check """
    return True


class Position:
    """Represents the internal state of a chess position"""

    def __init__(self, board):
        if board is None:
            self.board = Board()
        else:
            self.board = board

        self.to_move = Color.WHITE

        self.castle_rights = {Color.WHITE: True, Color.BLACK: True}
        self.en_passant_target = None  # target square
        self.halfmove_clock = 0

        self.piece_map = {}
        self.set_initial_piece_locations()

    def get_piece_locations(self):
        pass

    def set_initial_piece_locations(self):
        self.piece_map[Piece.wP] = set([i for i in range(8, 16)])
        self.piece_map[Piece.wR] = {0, 7}
        self.piece_map[Piece.wN] = {1, 6}
        self.piece_map[Piece.wB] = {2, 5}
        self.piece_map[Piece.wQ] = {3}
        self.piece_map[Piece.wK] = {4}

        self.piece_map[Piece.bP] = set([i for i in range(48, 56)])
        self.piece_map[Piece.bR] = {56, 63}
        self.piece_map[Piece.bN] = {57, 62}
        self.piece_map[Piece.bB] = {58, 61}
        self.piece_map[Piece.bQ] = {59}
        self.piece_map[Piece.bK] = {60}

    def make_move(self, move):
        if not is_legal_move(move):
            print('Illegal move')
            return

        # Empty piece existing spot
        self.piece_map[move.piece].remove(move.from_sq)

        # Put piece into new spot
        self.piece_map[move.piece].add(move.to_sq)

        # TODO: update
        self.castle_rights = self.castle_rights

        # TODO: update
        self.en_passant_target = self.en_passant_target

        self.halfmove_clock += 1

        # Data flows from MakeMove -> Position -> BitBoard -> Search
        self.board.update_bitboards(self.piece_map)

        self.to_move = not self.to_move

        return generate_fen()
