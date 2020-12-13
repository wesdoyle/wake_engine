import numpy as np

from wake.bitboard_helpers import set_bit, get_northwest_ray, bitscan_forward, get_northeast_ray, bitscan_reverse, \
    get_southwest_ray, get_southeast_ray, get_north_ray, get_east_ray, get_south_ray, \
    get_west_ray, make_uint64
from wake.board import Board
from wake.constants import Color, Piece, Rank, HOT, File
from wake.move import Move


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
        bitboard = make_uint64()
        move_square_bb = set_bit(bitboard, square)
        occupy_lookup = {
            Color.BLACK: self.board.black_pieces_bb,
            Color.WHITE: self.board.white_pieces_bb,
        }
        return occupy_lookup[self.color_to_move] & move_square_bb

    def intersects_with_opp_pieces(self, square):
        bitboard = make_uint64()
        move_square_bb = set_bit(bitboard, square)
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

        if move.piece in {Piece.wN, Piece.bN}:
            # if not in the static attack map return False
            return self.is_legal_knight_move(move)

        if move.piece in {Piece.wP, Piece.bP}:
            return self.is_legal_pawn_move(move)

        if move.piece in {Piece.wB, Piece.bB}:
            return self.is_legal_bishop_move(move)

        if move.piece in {Piece.wR, Piece.bR}:
            return self.is_legal_rook_move(move)

        if move.piece in {Piece.wQ, Piece.bQ}:
            return self.is_legal_queen_move(move)

        if move.piece in {Piece.wK, Piece.bK}:
            return self.is_legal_king_move(move)

        print("Uncaught illegal move")
        return False

    def is_legal_knight_move(self, move: Move) -> np.uint64:
        bitboard = make_uint64()
        legal_knight_moves = self.board.get_knight_attack_from(move.from_sq)
        if not legal_knight_moves & set_bit(bitboard, move.to_sq):
            return False
        # if intersects_with_own_pieces return False
        if self.intersects_with_own_pieces(move.to_sq):
            return False
        # if intersects_with_opp_pieces return True, is_capture => True
        if self.intersects_with_opp_pieces(move.to_sq):
            move.is_capture = True
            return True
        return True

    def is_legal_bishop_move(self, move: Move) -> np.uint64:
        """
        Pseudo-Legal Bishop Moves
        Implements the classical approach for determining legal sliding-piece moves
        for diagonal directions. Gets first blocker with forward or reverse bitscan
        based on the ray direction and XORs the open board ray with the ray continuation
        from the blocked square.
        :param move: the proposed Move instance
        :return: True iff Move is legal
        """
        bitboard = make_uint64()
        moving_to_square = set_bit(bitboard, move.to_sq)
        occupied = self.board.occupied_squares_bb

        # northwest route
        northwest_ray = get_northwest_ray(bitboard, move.from_sq)
        intersection = occupied & northwest_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northwest_ray(bitboard, first_blocker)
            northwest_ray ^= block_ray

        # northeast route
        northeast_ray = get_northeast_ray(bitboard, move.from_sq)
        intersection = occupied & northeast_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northeast_ray(bitboard, first_blocker)
            northeast_ray ^= block_ray

        # southwest route
        southwest_ray = get_southwest_ray(bitboard, move.from_sq)
        intersection = occupied & southwest_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southwest_ray(bitboard, first_blocker)
            southwest_ray ^= block_ray

        # southeast route
        southeast_ray = get_southeast_ray(bitboard, move.from_sq)
        intersection = occupied & southeast_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southeast_ray(bitboard, first_blocker)
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

    def is_legal_rook_move(self, move: Move) -> bool:
        """
        Pseudo-Legal Rook Moves
        Implements the classical approach for determining legal sliding-piece moves
        for rank and file directions. Gets first blocker with forward or reverse bitscan
        based on the ray direction and XORs the open board ray with the ray continuation
        from the blocked square.
        :param move: the proposed Move instance
        :return: True iff Move is legal
        """
        bitboard = make_uint64()
        moving_to_square = set_bit(bitboard, move.to_sq)
        occupied = self.board.occupied_squares_bb

        # north route
        north_ray = get_north_ray(bitboard, move.from_sq)
        intersection = occupied & north_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northwest_ray(bitboard, first_blocker)
            north_ray ^= block_ray

        # east route
        east_ray = get_east_ray(bitboard, move.from_sq)
        intersection = occupied & east_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northeast_ray(bitboard, first_blocker)
            east_ray ^= block_ray

        # south route
        south_ray = get_south_ray(bitboard, move.from_sq)
        intersection = occupied & south_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southwest_ray(bitboard, first_blocker)
            south_ray ^= block_ray

        # west route
        west_ray = get_west_ray(bitboard, move.from_sq)
        intersection = occupied & west_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southeast_ray(bitboard, first_blocker)
            west_ray ^= block_ray

        legal_moves = moving_to_square & (north_ray | east_ray | south_ray | west_ray)

        occupied_squares = {
            Color.BLACK: self.board.black_pieces_bb,
            Color.WHITE: self.board.white_pieces_bb,
        }

        # remove own piece targets
        own_piece_targets = occupied_squares[self.color_to_move] & moving_to_square
        if own_piece_targets:
            legal_moves &= ~own_piece_targets

        return legal_moves

    def is_legal_pawn_move(self, move: Move) -> bool:
        """
        Pseudo-Legal Pawn Moves:
        - Pawn non-attacks that don't intersect with occupied squares
        - Pawn attacks that intersect with opponent pieces
        :param move: the proposed Move instance
        :return: True iff Move is legal
        """
        bitboard = make_uint64()
        moving_to_square = set_bit(bitboard, move.to_sq)

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

        # Handle en-passant targets
        if self.en_passant_target:
            en_passant_bb = set_bit(bitboard, self.en_passant_target)
            en_passant_move = legal_attack_moves[self.color_to_move] & en_passant_bb
            legal_moves |= en_passant_move

        # Handle Captures
        if moving_to_square & legal_attack_moves[self.color_to_move]:
            move.is_capture = True

        # Handle removing own piece targets
        occupied_squares = {
            Color.BLACK: self.board.black_pieces_bb,
            Color.WHITE: self.board.white_pieces_bb,
        }

        own_piece_targets = occupied_squares[self.color_to_move] & moving_to_square

        if own_piece_targets:
            legal_moves &= ~own_piece_targets

        # Handle promotion
        promotion_rank = {
            Color.WHITE: Rank.hex8,
            Color.BLACK: Rank.hex1
        }

        if moving_to_square & promotion_rank[self.color_to_move]:
            move.is_promotion = True

        return legal_moves & moving_to_square

    def is_legal_queen_move(self, move):
        """
        Pseudo-Legal Queen Moves:  bitwise OR of legal Bishop moves, Rook moves
        :param move: the proposed Move instance
        :return: True iff Move is legal
        """
        return self.is_legal_rook_move(move) | self.is_legal_bishop_move(move)

    def is_legal_king_move(self, move):
        """
        Pseudo-Legal King Moves: one step in any direction
        :param move: the proposed Move instance
        :return: True iff Move is legal
        """
        bitboard = make_uint64()
        moving_to_square = set_bit(bitboard, move.to_sq)

        for i in [8, -8]:
            # North-South
            bitboard |= set_bit(bitboard, np.uint64(move.from_sq + i))
        for i in [1, 9, -7]:
            # East (mask the A file)
            bitboard |= set_bit(bitboard, np.uint64(move.from_sq + i) & ~np.uint64(File.hexA))
        for i in [-1, -9, 7]:
            # West (mask the H file)
            bitboard |= set_bit(bitboard, np.uint64(move.from_sq + i) & ~np.uint64(File.hexH))

        occupied_squares = {
            Color.BLACK: self.board.black_pieces_bb,
            Color.WHITE: self.board.white_pieces_bb,
        }

        # remove own piece targets
        own_piece_targets = occupied_squares[self.color_to_move] & moving_to_square

        if own_piece_targets:
            bitboard &= ~own_piece_targets

        return bitboard
