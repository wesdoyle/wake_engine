import numpy as np

from wake.bitboard_helpers import set_bit, get_northwest_ray, bitscan_forward, get_northeast_ray, bitscan_reverse, \
    get_southwest_ray, get_southeast_ray, get_north_ray, get_east_ray, get_south_ray, \
    get_west_ray, make_uint64
from wake.board import Board
from wake.constants import Color, Piece, Rank, File, Square
from wake.move import Move


# TODO: possible side-effects from mutating move all over
#  the place in move legality checking

def generate_fen():
    """ TODO: Generate FEN for the current state """
    return '-- TODO: generate FEN --'


class Position:
    """
    Represents the internal state of a chess position
    """

    def __init__(self, board=None):
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

    @property
    def get_occupied_squares_by_color(self):
        return {
            Color.BLACK: self.board.black_pieces_bb,
            Color.WHITE: self.board.white_pieces_bb,
        }

    # -------------------------------------------------------------
    # MAKE MOVE
    # -------------------------------------------------------------

    def make_move(self, move):
        if not self.is_legal_move(move):
            return

        self.piece_map[move.piece].remove(move.from_sq)
        self.piece_map[move.piece].add(move.to_sq)
        self.halfmove_clock += 1

        self.board.update_bitboards(self.piece_map)

        self.color_to_move = not self.color_to_move
        return generate_fen()

    # -------------------------------------------------------------
    # MOVE LEGALITY CHECKING
    # -------------------------------------------------------------

    def is_legal_move(self, move: Move) -> bool:

        if self.is_wrong_color_piece(move):
            return False

        if move.piece in {Piece.wN, Piece.bN}:
            return self.get_legal_knight_moves_from(move).any()

        if move.piece in {Piece.wP, Piece.bP}:
            return self.get_legal_pawn_moves_from(move).any()

        if move.piece in {Piece.wB, Piece.bB}:
            return self.get_legal_bishop_moves_from(move).any()

        if move.piece in {Piece.wR, Piece.bR}:
            return self.get_legal_rook_moves_from(move).any()

        if move.piece in {Piece.wQ, Piece.bQ}:
            return self.get_legal_queen_moves_from(move).any()

        if move.piece in {Piece.wK, Piece.bK}:
            return self.get_legal_king_moves_from(move).any()

        print("Uncaught illegal move")
        return False

    def is_wrong_color_piece(self, move):
        if self.color_to_move == Color.WHITE and move.piece not in Piece.white_pieces:
            print("That isn't one of your pieces.")
            return False

        if self.color_to_move == Color.BLACK and move.piece not in Piece.black_pieces:
            print("That isn't one of your pieces.")
            return False

    def get_current_attacked_squares(self, move: Move) -> np.uint64:
        return self.get_legal_pawn_moves_from(move) \
           | self.get_legal_knight_moves_from(move) \
           | get_legal_bishop_moves_from(move) \
           | get_legal_rook_moves_from(move) \
           | get_legal_queen_moves_from(move) \
           | get_legal_king_moves_from(move)

    def get_white_attacked_squares(self, move: Move) -> np.uint64:
        return self.get_legal_pawn_moves_from(move) \
               | self.get_legal_knight_moves_from(move) \
               | get_legal_bishop_moves_from(move) \
               | get_legal_rook_moves_from(move) \
               | get_legal_queen_moves_from(move) \
               | get_legal_king_moves_from(move)

    def get_black_attacked_squares(self, move: Move) -> np.uint64:
        return self.get_legal_pawn_moves_from(move) \
               | self.get_legal_knight_moves_from(move) \
               | get_legal_bishop_moves_from(move) \
               | get_legal_rook_moves_from(move) \
               | get_legal_queen_moves_from(move) \
               | get_legal_king_moves_from(move)

    # -------------------------------------------------------------
    # LEGAL KNIGHT MOVES
    # -------------------------------------------------------------

    def get_legal_knight_moves_from(self, move: Move, color_to_move: int) -> np.uint64:
        """
        Gets the legal knight moves from the given Move instance
        :param move:
        :param color_to_move:
        :return:
        """
        legal_knight_moves = self.board.get_knight_attack_from(move.from_sq)

        # Mask out own pieces
        if color_to_move == Color.WHITE:
            legal_knight_moves &= ~self.board.white_pieces_bb

        if color_to_move == Color.BLACK:
            legal_knight_moves &= ~self.board.black_pieces_bb

        return legal_knight_moves

    # -------------------------------------------------------------
    # LEGAL BISHOP MOVES
    # -------------------------------------------------------------

    def get_legal_bishop_moves_from(self, move: Move, color_to_move: int) -> np.uint64:
        """
        Pseudo-Legal Bishop Moves
        Implements the classical approach for determining legal sliding-piece moves
        for diagonal directions. Gets first blocker with forward or reverse bitscan
        based on the ray direction and XORs the open board ray with the ray continuation
        from the blocked square.
        :param move: the proposed Move instance
        :param color_to_move: the current color to move
        :return: True iff Move is legal
        """
        bitboard = make_uint64()
        moving_to_square = set_bit(bitboard, move.to_sq)
        occupied = self.board.occupied_squares_bb

        northwest_ray = get_northwest_ray(bitboard, move.from_sq)
        intersection = occupied & northwest_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northwest_ray(bitboard, first_blocker)
            northwest_ray ^= block_ray

        northeast_ray = get_northeast_ray(bitboard, move.from_sq)
        intersection = occupied & northeast_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northeast_ray(bitboard, first_blocker)
            northeast_ray ^= block_ray

        southwest_ray = get_southwest_ray(bitboard, move.from_sq)
        intersection = occupied & southwest_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southwest_ray(bitboard, first_blocker)
            southwest_ray ^= block_ray

        southeast_ray = get_southeast_ray(bitboard, move.from_sq)
        intersection = occupied & southeast_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southeast_ray(bitboard, first_blocker)
            southeast_ray ^= block_ray

        legal_moves = moving_to_square & (northwest_ray | northeast_ray | southwest_ray | southeast_ray)

        # remove own piece targets
        own_piece_targets = self.get_occupied_squares_by_color[color_to_move] & moving_to_square
        if own_piece_targets:
            legal_moves &= ~own_piece_targets

        return legal_moves

    # -------------------------------------------------------------
    # LEGAL ROOK MOVES
    # -------------------------------------------------------------

    def get_legal_rook_moves_from(self, move: Move, color_to_move: int) -> np.uint64:
        """
        Pseudo-Legal Rook Moves
        Implements the classical approach for determining legal sliding-piece moves
        for rank and file directions. Gets first blocker with forward or reverse bitscan
        based on the ray direction and XORs the open board ray with the ray continuation
        from the blocked square.
        :param move: the proposed Move instance
        :param color_to_move: the current color to move
        :return: True iff Move is legal
        """
        bitboard = make_uint64()
        moving_to_square = set_bit(bitboard, move.to_sq)
        occupied = self.board.occupied_squares_bb

        north_ray = get_north_ray(bitboard, move.from_sq)
        intersection = occupied & north_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northwest_ray(bitboard, first_blocker)
            north_ray ^= block_ray

        east_ray = get_east_ray(bitboard, move.from_sq)
        intersection = occupied & east_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northeast_ray(bitboard, first_blocker)
            east_ray ^= block_ray

        south_ray = get_south_ray(bitboard, move.from_sq)
        intersection = occupied & south_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southwest_ray(bitboard, first_blocker)
            south_ray ^= block_ray

        west_ray = get_west_ray(bitboard, move.from_sq)
        intersection = occupied & west_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southeast_ray(bitboard, first_blocker)
            west_ray ^= block_ray

        legal_moves = north_ray | east_ray | south_ray | west_ray

        # Remove own piece targets
        own_piece_targets = self.get_occupied_squares_by_color[color_to_move] & moving_to_square

        if own_piece_targets:
            legal_moves &= ~own_piece_targets

        return legal_moves

    def get_legal_pawn_moves_from(self, move: Move, color_to_move: int) -> np.uint64:
        """
        Pseudo-Legal Pawn Moves:
        - Pawn non-attacks that don't intersect with occupied squares
        - Pawn attacks that intersect with opponent pieces
        :param move: the proposed Move instance
        :param color_to_move: the current color to move
        :return: True iff Move is legal
        """
        bitboard = make_uint64()
        moving_to_square = set_bit(bitboard, move.to_sq)

        legal_non_attack_moves = {
            Color.WHITE: self.board.white_pawn_motion_bbs[move.from_sq],
            Color.BLACK: self.board.black_pawn_motion_bbs[move.from_sq]
        }

        legal_non_attack_moves[color_to_move] &= self.board.empty_squares_bb

        legal_attack_moves = {
            Color.WHITE: self.board.white_pawn_attack_bbs[move.from_sq],
            Color.BLACK: self.board.black_pawn_attack_bbs[move.from_sq]
        }

        opp_occupied = {
            Color.WHITE: self.board.black_pieces_bb,
            Color.BLACK: self.board.white_pieces_bb
        }

        legal_attack_moves[color_to_move] &= opp_occupied[color_to_move]

        legal_moves = legal_non_attack_moves[color_to_move] | legal_attack_moves[color_to_move]

        # Handle en-passant targets
        if self.en_passant_target:
            en_passant_bb = set_bit(bitboard, self.en_passant_target)
            en_passant_move = legal_attack_moves[color_to_move] & en_passant_bb
            legal_moves |= en_passant_move

        # Handle Captures
        if moving_to_square & legal_attack_moves[color_to_move]:
            move.is_capture = True

        # Handle removing own piece targets
        occupied_squares = {
            Color.BLACK: self.board.black_pieces_bb,
            Color.WHITE: self.board.white_pieces_bb,
        }

        # Remove own piece targets
        own_piece_targets = occupied_squares[color_to_move] & moving_to_square

        if own_piece_targets:
            legal_moves &= ~own_piece_targets

        # Handle promotion
        promotion_rank = {
            Color.WHITE: Rank.hex8,
            Color.BLACK: Rank.hex1
        }

        if moving_to_square & promotion_rank[color_to_move]:
            move.is_promotion = True

        return legal_moves

    def get_legal_queen_moves_from(self, move: Move, color_to_move: int) -> np.uint64:
        """
        Pseudo-Legal Queen Moves:  bitwise OR of legal Bishop moves, Rook moves
        :param move: the proposed Move instance
        :param color_to_move: the current color to move
        :return: True iff Move is legal
        """
        return self.get_legal_rook_moves_from(move) | self.get_legal_bishop_moves_from(move)

    def get_legal_king_moves_from(self, move: Move, color_to_move: int) -> np.uint64:
        """
        Pseudo-Legal King Moves: one step in any direction
        :param move: the proposed Move instance
        :param color_to_move: the current color to move
        :return: True iff Move is legal
        """
        bitboard = make_uint64()
        moving_to_square = set_bit(bitboard, move.to_sq)

        # Handle Castling
        if move.is_castling:
            if self.can_castle():
                bitboard = self.add_castling_moves(bitboard)

        for i in [8, -8]:
            # North-South
            bitboard |= set_bit(bitboard, np.uint64(move.from_sq + i))
        for i in [1, 9, -7]:
            # East (mask the A file)
            bitboard |= set_bit(bitboard, np.uint64(move.from_sq + i) & ~np.uint64(File.hexA))
        for i in [-1, -9, 7]:
            # West (mask the H file)
            bitboard |= set_bit(bitboard, np.uint64(move.from_sq + i) & ~np.uint64(File.hexH))

        # remove own piece targets
        own_piece_targets = self.get_occupied_squares_by_color[color_to_move] & moving_to_square

        if own_piece_targets:
            bitboard &= ~own_piece_targets

        return bitboard

    def add_castling_moves(self, bitboard: np.uint64, color_to_move: int) -> np.uint64:
        """
        Adds castling squares to the bitboard
        :param bitboard: numpy uint64 bitboard
        :param color_to_move: the current color to move
        :return:
        """
        if color_to_move == Color.WHITE:
            bitboard |= set_bit(bitboard, Square.C1)
            bitboard |= set_bit(bitboard, Square.G1)
        if color_to_move == Color.BLACK:
            bitboard |= set_bit(bitboard, Square.C8)
            bitboard |= set_bit(bitboard, Square.G8)
        return bitboard

    def can_castle(self, color_to_move) -> tuple:
        """
        Returns a tuple of (bool, bool) can castle on (kingside, queenside)
        and can move through the castling squares without being in check.
        :return: Tuple (bool, bool) iff the color_to_move has castling rights
        and can move through the castling squares without being in check on (kingside, queenside)
        """
        if not self.castle_rights[color_to_move]:
            return 0, 0

        if color_to_move == Color.WHITE:
            self.squares_attacked_by_black & self.castle_route
            return 0, 0

        if color_to_move == Color.BLACK:
            self.squares_attacked_by_white & self.castle_route
            return 0, 0
