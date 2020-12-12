from bitboard_helpers import set_bit
from board import Board
from constants import Color, Piece, Rank
from move import Move
import numpy as np


# TODO: possible side-effects from mutating move all over
#  the place in move legality checking

# TODO: On illegal move, it looks like we're resetting to starting position

# TODO: Bugfix pawn motion after first pawn move

def generate_fen():
    """ TODO: Generate FEN for the current state """
    return '-- TODO: generate FEN --'


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
        if not self.is_legal_move(move):
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

    def intersects_with_own_pieces(self, square):
        bb = np.uint64(0)
        move_square_bb = set_bit(bb, square)
        occupy_lookup = {
            Color.BLACK: self.board.black_pieces_bb,
            Color.WHITE: self.board.white_pieces_bb,
        }
        return occupy_lookup[self.to_move] & move_square_bb

    def intersects_with_opp_pieces(self, square):
        bb = np.uint64(0)
        move_square_bb = set_bit(bb, square)
        occupy_lookup = {
            Color.WHITE: self.board.black_pieces_bb,
            Color.BLACK: self.board.white_pieces_bb,
        }
        return occupy_lookup[self.to_move] & move_square_bb

    def is_legal_move(self, move: Move) -> bool:
        if self.to_move == Color.WHITE and move.piece not in Piece.white_pieces:
            print("That isn't one of your pieces.")
            return False

        if self.to_move == Color.BLACK and move.piece not in Piece.black_pieces:
            print("That isn't one of your pieces.")
            return False

        # Get position of piece from piece map
        # If its a knight, lookup if move.to & with the knight attack bb
        bb = np.uint64(0)

        if move.piece in {Piece.wN, Piece.bN}:
            # if not in the static attack map return False
            return self.is_legal_knight_move(move, bb)

        if move.piece in {Piece.wP,  Piece.bP}:
            return self.is_legal_pawn_move(move, bb)

        print("Uncaught illegal move")
        return False

        # If its a king, lookup if move.to & with the knight attack bb

        # If its sliding, lookup if move.to &:
        # if own, square up to bni bit-scanned find in the ray direction
        # if opp, square up to inc bit-scanned find in the ray direction, capture

    def is_legal_knight_move(self, move, bb):
        legal_knight_moves = self.board.get_knight_attack_from(move.from_sq)
        if not legal_knight_moves & set_bit(bb, move.to_sq):
            return False
        # if intersects_with_own_pieces return False
        if self.intersects_with_own_pieces(move.to_sq):
            return False
        # if intersects_with_opp_pieces return True, is_capture => True
        if self.intersects_with_opp_pieces(move.to_sq):
            move.is_capture = True
            return True
        return True

    def is_legal_pawn_move(self, move, bb):
        """
        Legal Pawn Moves:
        - Pawn non-attacks that don't intersect with occupied squares
        - Pawn attacks that intersect with opponent pieces
        :param move: The proposed move
        :param bb: An empty bitboard
        :return: (bool) is a legal pawn move
        """
        moving_to_square = set_bit(bb, move.to_sq)

        legal_non_attack_moves = {
            Color.WHITE: self.board.white_pawn_motion_bbs[move.from_sq],
            Color.BLACK: self.board.black_pawn_motion_bbs[move.from_sq]
        }

        legal_non_attack_moves[self.to_move] &= self.board.empty_squares_bb

        legal_attack_moves = {
            Color.WHITE: self.board.white_pawn_attack_bbs[move.from_sq],
            Color.BLACK: self.board.black_pawn_attack_bbs[move.from_sq]
        }

        opp_occupied = {
            Color.WHITE: self.board.black_pieces_bb,
            Color.BLACK: self.board.white_pieces_bb
        }

        legal_attack_moves[self.to_move] &= opp_occupied[self.to_move]

        legal_moves = legal_non_attack_moves[self.to_move] | legal_attack_moves[self.to_move]

        if self.en_passant_target:
            en_passant_bb = set_bit(bb, self.en_passant_target)
            en_passant_move = legal_attack_moves[self.to_move] & en_passant_bb
            legal_moves |= en_passant_move

        if moving_to_square & legal_attack_moves[self.to_move]:
            move.is_capture = True

        promotion_rank = {
            Color.WHITE: Rank.hex8,
            Color.BLACK: Rank.hex1
        }

        if moving_to_square & promotion_rank[self.to_move]:
            move.is_promotion = True

        print("legal moves:", legal_moves)
        print("moving_to_square:", moving_to_square)
        print("legal pawn move:", legal_moves & moving_to_square)
        return legal_moves & moving_to_square
