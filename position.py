from bitboard_helpers import set_bit, get_northwest_ray, bitscan_forward, get_northeast_ray, bitscan_reverse, \
    get_southwest_ray, get_southeast_ray, make_uint64, clear_bit
from board import Board
from constants import Color, Piece, Rank
from move import Move
import numpy as np


# TODO: possible side-effects from mutating move all over
#  the place in move legality checking

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

        self.color_to_move = Color.WHITE

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

        self.color_to_move = not self.color_to_move

        return generate_fen()

    def intersects_with_own_pieces(self, square):
        bb = np.uint64(0)
        move_square_bb = set_bit(bb, square)
        occupy_lookup = {
            Color.BLACK: self.board.black_pieces_bb,
            Color.WHITE: self.board.white_pieces_bb,
        }
        return occupy_lookup[self.color_to_move] & move_square_bb

    def intersects_with_opp_pieces(self, square):
        bb = np.uint64(0)
        move_square_bb = set_bit(bb, square)
        occupy_lookup = {
            Color.WHITE: self.board.black_pieces_bb,
            Color.BLACK: self.board.white_pieces_bb,
        }
        return occupy_lookup[self.color_to_move] & move_square_bb

    def is_legal_move(self, move: Move) -> bool:
        if self.color_to_move == Color.WHITE and move.piece not in Piece.white_pieces:
            print("That isn't one of your pieces.")
            return False

        if self.color_to_move == Color.BLACK and move.piece not in Piece.black_pieces:
            print("That isn't one of your pieces.")
            return False

        # Get position of piece from piece map
        # If its a knight, lookup if move.to & with the knight attack bb
        bb = np.uint64(0)

        if move.piece in {Piece.wN, Piece.bN}:
            # if not in the static attack map return False
            return self.is_legal_knight_move(move, bb)

        if move.piece in {Piece.wP, Piece.bP}:
            return self.is_legal_pawn_move(move, bb)

        if move.piece in {Piece.wP, Piece.bP}:
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

    def is_legal_bishop_move(self, move, bb):
        """
        Implements the classical approach for determining legal sliding-piece moves
        for diagonal directions. Gets first blocker with forward or reverse bitscan
        based on the ray direction and XORs the open board ray with the ray continuation
        from the blocked square.
        :param move:
        :param bb:
        :return:
        """
        moving_to_square = set_bit(bb, move.to_sq)

        # northwest attack route
        occupied = self.board.occupied_squares_bb
        northwest_ray = get_northwest_ray(bb, move.from_sq)
        intersection = occupied & northwest_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northwest_ray(bb, first_blocker)
            northwest_ray ^= block_ray

        # northeast route
        occupied = self.board.occupied_squares_bb
        northeast_ray = get_northeast_ray(bb, move.from_sq)
        intersection = occupied & northeast_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northeast_ray(bb, first_blocker)
            northeast_ray ^= block_ray

        # southwest route
        occupied = self.board.occupied_squares_bb
        southwest_ray = get_southwest_ray(bb, move.from_sq)
        intersection = occupied & southwest_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southwest_ray(bb, first_blocker)
            southwest_ray ^= block_ray

        # southeast route
        occupied = self.board.occupied_squares_bb
        southeast_ray = get_southeast_ray(bb, move.from_sq)
        intersection = occupied & southeast_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southeast_ray(bb, first_blocker)
            southeast_ray ^= block_ray

        occupied_squares = {
            Color.BLACK: self.board.black_pieces_bb,
            Color.WHITE: self.board.white_pieces_bb,
        }

        legal_moves = moving_to_square & (northwest_ray | northeast_ray | southwest_ray | southeast_ray)

        # remove own piece targets
        own_piece_targets = occupied_squares[self.color_to_move] & moving_to_square
        if own_piece_targets:
            legal_moves &= ~own_piece_targets

        return legal_moves

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

        legal_non_attack_moves[self.color_to_move] &= self.board.empty_squares_bb

        legal_attack_moves = {
            Color.WHITE: self.board.white_pawn_attack_bbs[move.from_sq],
            Color.BLACK: self.board.black_pawn_attack_bbs[move.from_sq]
        }

        opp_occupied = {
            Color.WHITE: self.board.black_pieces_bb,
            Color.BLACK: self.board.white_pieces_bb
        }

        legal_attack_moves[self.color_to_move] &= opp_occupied[self.color_to_move]

        legal_moves = legal_non_attack_moves[self.color_to_move] | legal_attack_moves[self.color_to_move]

        if self.en_passant_target:
            en_passant_bb = set_bit(bb, self.en_passant_target)
            en_passant_move = legal_attack_moves[self.color_to_move] & en_passant_bb
            legal_moves |= en_passant_move

        if moving_to_square & legal_attack_moves[self.color_to_move]:
            move.is_capture = True

        promotion_rank = {
            Color.WHITE: Rank.hex8,
            Color.BLACK: Rank.hex1
        }

        if moving_to_square & promotion_rank[self.color_to_move]:
            move.is_promotion = True

        print("legal moves:", legal_moves)
        print("moving_to_square:", moving_to_square)
        print("legal pawn move:", legal_moves & moving_to_square)
        return legal_moves & moving_to_square
