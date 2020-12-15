import numpy as np

from wake.bitboard_helpers import set_bit, get_northwest_ray, bitscan_forward, get_northeast_ray, bitscan_reverse, \
    get_southwest_ray, get_southeast_ray, get_north_ray, get_east_ray, get_south_ray, \
    get_west_ray, make_uint64, pprint_bb
from wake.board import Board
from wake.constants import Color, Piece, File, Square, CastleRoute
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

        self.white_pawn_moves = make_uint64()
        self.white_pawn_attacks = make_uint64()
        self.white_rook_attacks = make_uint64()
        self.white_knight_attacks = make_uint64()
        self.white_bishop_attacks = make_uint64()
        self.white_queen_attacks = make_uint64()
        self.white_king_attacks = make_uint64()

        self.black_pawn_moves = make_uint64()
        self.black_pawn_attacks = make_uint64()
        self.black_rook_attacks = make_uint64()
        self.black_knight_attacks = make_uint64()
        self.black_bishop_attacks = make_uint64()
        self.black_queen_attacks = make_uint64()
        self.black_king_attacks = make_uint64()

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

    @property
    def black_attacked_squares(self):
        return self.black_rook_attacks \
               | self.black_bishop_attacks \
               | self.black_knight_attacks \
               | self.black_pawn_attacks \
               | self.black_queen_attacks \
               | self.black_king_attacks

    @property
    def white_attacked_squares(self):
        return self.white_rook_attacks \
               | self.white_bishop_attacks \
               | self.white_knight_attacks \
               | self.white_pawn_attacks \
               | self.white_queen_attacks \
               | self.white_king_attacks

    # -------------------------------------------------------------
    # MAKE MOVE
    # -------------------------------------------------------------

    def make_move(self, move):
        if not self.is_legal_move(move):
            print("Illegal move")
            return

        self.piece_map[move.piece].remove(move.from_sq)
        self.piece_map[move.piece].add(move.to_sq)
        self.halfmove_clock += 1

        self.board.update_bitboards(self.piece_map)

        # TODO: Decide what's a function; move what belongs to bb updates there

        self.white_rook_attacks = make_uint64()
        self.black_rook_attacks = make_uint64()
        self.white_bishop_attacks = make_uint64()
        self.black_bishop_attacks = make_uint64()
        self.white_knight_attacks = make_uint64()
        self.black_knight_attacks = make_uint64()
        self.white_queen_attacks = make_uint64()
        self.black_queen_attacks = make_uint64()
        self.white_king_attacks = make_uint64()
        self.black_king_attacks = make_uint64()

        for piece, squares in self.piece_map.items():
            if piece == Piece.wP:
                for square in squares:
                    self.update_legal_pawn_moves(square, Color.WHITE)
            if piece == Piece.bP:
                for square in squares:
                    self.update_legal_pawn_moves(square, Color.BLACK)

            if piece == Piece.wR:
                for square in squares:
                    self.update_legal_rook_moves(square, Color.WHITE)
            if piece == Piece.bR:
                for square in squares:
                    self.update_legal_rook_moves(square, Color.BLACK)

            if piece == Piece.wB:
                for square in squares:
                    self.update_legal_bishop_moves(square, Color.WHITE)
            if piece == Piece.bB:
                for square in squares:
                    self.update_legal_bishop_moves(square, Color.BLACK)

            if piece == Piece.wN:
                for square in squares:
                    self.update_legal_knight_moves(square, Color.WHITE)
            if piece == Piece.bN:
                for square in squares:
                    self.update_legal_knight_moves(square, Color.BLACK)

            if piece == Piece.wQ:
                for square in squares:
                    self.update_legal_queen_moves(square, Color.WHITE)
            if piece == Piece.bQ:
                for square in squares:
                    self.update_legal_queen_moves(square, Color.BLACK)

            if piece == Piece.wK:
                for square in squares:
                    self.update_legal_king_moves(square, Color.WHITE)
            if piece == Piece.bK:
                for square in squares:
                    self.update_legal_king_moves(square, Color.BLACK)

        self.color_to_move = not self.color_to_move
        return generate_fen()

    # -------------------------------------------------------------
    # MOVE LEGALITY CHECKING
    # -------------------------------------------------------------

    def is_legal_move(self, move: Move) -> bool:
        if move.piece in (Piece.wP, Piece.bP):
            return self.is_legal_pawn_move(move)
        if move.piece in (Piece.wB, Piece.bB):
            return self.is_legal_bishop_move(move)
        if move.piece in (Piece.wR, Piece.bR):
            return self.is_legal_rook_move()
        if move.piece in (Piece.wN, Piece.bN):
            return self.is_legal_knight_move(move)
        if move.piece in (Piece.wQ, Piece.bQ):
            return self.is_legal_queen_move(move)
        if move.piece in (Piece.wK, Piece.bK):
            return self.is_legal_king_move(move)

    def is_not_pawn_motion_or_attack(self, to_sq):
        moving_to_square_bb = set_bit(make_uint64(), to_sq)
        if self.color_to_move == Color.WHITE:
            if not (self.board.white_pawn_attack_bbs[to_sq]
                    | self.board.white_pawn_attack_bbs[to_sq]) & moving_to_square_bb:
                return False
        if self.color_to_move == Color.BLACK:
            if not (self.board.black_pawn_attack_bbs[to_sq]
                    | self.board.black_pawn_attack_bbs[to_sq]) & moving_to_square_bb:
                return False

    def is_legal_pawn_move(self, move: Move) -> bool:
        current_square_bb = set_bit(make_uint64(), move.from_sq)
        moving_to_square_bb = set_bit(make_uint64(), move.to_sq)

        if move.piece == Piece.wP:
            if not (self.board.white_P_bb & current_square_bb):
                return False
            if self.is_not_pawn_motion_or_attack(move.to_sq):
                return False
            if (self.white_pawn_attacks & moving_to_square_bb) & ~self.board.black_pieces_bb:
                return False
            return True

        if move.piece == Piece.bP:
            if not (self.board.black_P_bb & current_square_bb):
                return False
            if not (self.board.black_pawn_motion_bbs[move.from_sq] & move.to_sq) \
                    or (self.board.black_pawn_attack_bbs[move.from_sq] & move.to_sq):
                return False
            return True

    def is_legal_bishop_move(self, move):
        return True

    def is_legal_rook_move(self, move):
        return True

    def is_legal_knight_move(self, move):
        return True

    def is_legal_queen_move(self, move):
        return True

    def is_legal_king_move(self, move):
        return True

    def is_wrong_color_piece(self, move):
        if self.color_to_move == Color.WHITE and move.piece not in Piece.white_pieces:
            return False

        if self.color_to_move == Color.BLACK and move.piece not in Piece.black_pieces:
            return False

    # -------------------------------------------------------------
    # LEGAL KNIGHT MOVES
    # -------------------------------------------------------------

    def update_legal_knight_moves(self, move_from_sq: np.uint64, color_to_move: int) -> np.uint64:
        """
        Gets the legal knight moves from the given Move instance
        :param move_from_sq:
        :param color_to_move:
        :return:
        """
        legal_knight_moves = self.board.get_knight_attack_from(move_from_sq)

        # Mask out own pieces
        if color_to_move == Color.WHITE:
            legal_knight_moves &= ~self.board.white_pieces_bb
            self.white_knight_attacks |= legal_knight_moves

        if color_to_move == Color.BLACK:
            legal_knight_moves &= ~self.board.black_pieces_bb
            self.black_knight_attacks |= legal_knight_moves

        return legal_knight_moves

    # -------------------------------------------------------------
    # LEGAL BISHOP MOVES
    # -------------------------------------------------------------

    def update_legal_bishop_moves(self, move_from_square: np.uint64, color_to_move: int) -> np.uint64:
        """
        Pseudo-Legal Bishop Moves
        Implements the classical approach for determining legal sliding-piece moves
        for diagonal directions. Gets first blocker with forward or reverse bitscan
        based on the ray direction and XORs the open board ray with the ray continuation
        from the blocked square.
        :param move_from_square: the proposed square from which the bishop is to move
        :param color_to_move: the current color to move
        :return: True iff Move is legal
        """
        bitboard = make_uint64()
        occupied = self.board.occupied_squares_bb

        northwest_ray = get_northwest_ray(bitboard, move_from_square)
        intersection = occupied & northwest_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northwest_ray(bitboard, first_blocker)
            northwest_ray ^= block_ray

        northeast_ray = get_northeast_ray(bitboard, move_from_square)
        intersection = occupied & northeast_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northeast_ray(bitboard, first_blocker)
            northeast_ray ^= block_ray

        southwest_ray = get_southwest_ray(bitboard, move_from_square)
        intersection = occupied & southwest_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southwest_ray(bitboard, first_blocker)
            southwest_ray ^= block_ray

        southeast_ray = get_southeast_ray(bitboard, move_from_square)
        intersection = occupied & southeast_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southeast_ray(bitboard, first_blocker)
            southeast_ray ^= block_ray

        legal_moves = northwest_ray | northeast_ray | southwest_ray | southeast_ray

        # remove own piece targets
        own_piece_targets = self.get_occupied_squares_by_color[color_to_move]
        if own_piece_targets:
            legal_moves &= ~own_piece_targets

        if color_to_move == Color.WHITE:
            self.white_bishop_attacks |= legal_moves

        if color_to_move == Color.BLACK:
            self.black_bishop_attacks |= legal_moves

        return legal_moves

    # -------------------------------------------------------------
    # LEGAL ROOK MOVES
    # -------------------------------------------------------------

    def update_legal_rook_moves(self, move_from_square: np.uint64, color_to_move: int) -> np.uint64:
        """
        Pseudo-Legal Rook Moves
        Implements the classical approach for determining legal sliding-piece moves
        for rank and file directions. Gets first blocker with forward or reverse bitscan
        based on the ray direction and XORs the open board ray with the ray continuation
        from the blocked square.
        :param move_from_square: the proposed square from which the rook is to move
        :param color_to_move: the current color to move
        :return: True iff Move is legal
        """
        bitboard = make_uint64()
        occupied = self.board.occupied_squares_bb

        north_ray = get_north_ray(bitboard, move_from_square)
        intersection = occupied & north_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_north_ray(bitboard, first_blocker)
            north_ray ^= block_ray

        east_ray = get_east_ray(bitboard, move_from_square)
        intersection = occupied & east_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_east_ray(bitboard, first_blocker)
            east_ray ^= block_ray

        south_ray = get_south_ray(bitboard, move_from_square)
        intersection = occupied & south_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_south_ray(bitboard, first_blocker)
            south_ray ^= block_ray

        west_ray = get_west_ray(bitboard, move_from_square)
        intersection = occupied & west_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_west_ray(bitboard, first_blocker)
            west_ray ^= block_ray

        legal_moves = north_ray | east_ray | south_ray | west_ray

        # Remove own piece targets
        own_piece_targets = self.get_occupied_squares_by_color[color_to_move]

        if own_piece_targets:
            legal_moves &= ~own_piece_targets

        if color_to_move == Color.WHITE:
            self.white_rook_attacks |= legal_moves

        if color_to_move == Color.BLACK:
            self.black_rook_attacks |= legal_moves

    def update_legal_pawn_moves(self, move_from_square: np.uint64, color_to_move: int):
        """
        Pseudo-Legal Pawn Moves:
        - Pawn non-attacks that don't intersect with occupied squares
        - Pawn attacks that intersect with opponent pieces
        :param move_from_square: the proposed square from which the pawn is to move
        :param color_to_move: the current color to move
        :return: True iff Move is legal
        """
        bitboard = make_uint64()

        legal_non_attack_moves = {
            Color.WHITE: self.board.white_pawn_motion_bbs[move_from_square],
            Color.BLACK: self.board.black_pawn_motion_bbs[move_from_square]
        }

        legal_non_attack_moves[color_to_move] &= self.board.empty_squares_bb

        legal_attack_moves = {
            Color.WHITE: self.board.white_pawn_attack_bbs[move_from_square],
            Color.BLACK: self.board.black_pawn_attack_bbs[move_from_square]
        }

        # Handle en-passant targets
        if self.en_passant_target:
            en_passant_bb = set_bit(bitboard, self.en_passant_target)
            en_passant_move = legal_attack_moves[color_to_move] & en_passant_bb
            if en_passant_move:
                legal_attack_moves[color_to_move] |= en_passant_move

        legal_moves = legal_non_attack_moves[color_to_move] | legal_attack_moves[color_to_move]

        # Handle removing own piece targets
        occupied_squares = {
            Color.BLACK: self.board.black_pieces_bb,
            Color.WHITE: self.board.white_pieces_bb,
        }

        # Remove own piece targets
        own_piece_targets = occupied_squares[color_to_move]

        if own_piece_targets:
            legal_moves &= ~own_piece_targets

        if color_to_move == Color.WHITE:
            self.white_pawn_attacks |= legal_attack_moves[color_to_move]
            self.white_pawn_moves |= legal_non_attack_moves[color_to_move]

        if color_to_move == Color.BLACK:
            self.black_pawn_attacks |= legal_attack_moves[color_to_move]
            self.black_pawn_moves |= legal_non_attack_moves[color_to_move]

    def update_legal_queen_moves(self, move_from_square: np.uint64, color_to_move: int):
        """
        Pseudo-Legal Queen Moves:  bitwise OR of legal Bishop moves, Rook moves
        :param move_from_square: the proposed square from which the queen is to move
        :param color_to_move: the current color to move
        :return: True iff Move is legal
        """

        # TODO: Reduce duplication

        bitboard = make_uint64()
        occupied = self.board.occupied_squares_bb

        north_ray = get_north_ray(bitboard, move_from_square)
        intersection = occupied & north_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_north_ray(bitboard, first_blocker)
            north_ray ^= block_ray

        east_ray = get_east_ray(bitboard, move_from_square)
        intersection = occupied & east_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_east_ray(bitboard, first_blocker)
            east_ray ^= block_ray

        south_ray = get_south_ray(bitboard, move_from_square)
        intersection = occupied & south_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_south_ray(bitboard, first_blocker)
            south_ray ^= block_ray

        west_ray = get_west_ray(bitboard, move_from_square)
        intersection = occupied & west_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_west_ray(bitboard, first_blocker)
            west_ray ^= block_ray

        northwest_ray = get_northwest_ray(bitboard, move_from_square)
        intersection = occupied & northwest_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northwest_ray(bitboard, first_blocker)
            northwest_ray ^= block_ray

        northeast_ray = get_northeast_ray(bitboard, move_from_square)
        intersection = occupied & northeast_ray
        if intersection:
            first_blocker = bitscan_forward(intersection)
            block_ray = get_northeast_ray(bitboard, first_blocker)
            northeast_ray ^= block_ray

        southwest_ray = get_southwest_ray(bitboard, move_from_square)
        intersection = occupied & southwest_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southwest_ray(bitboard, first_blocker)
            southwest_ray ^= block_ray

        southeast_ray = get_southeast_ray(bitboard, move_from_square)
        intersection = occupied & southeast_ray
        if intersection:
            first_blocker = bitscan_reverse(intersection)
            block_ray = get_southeast_ray(bitboard, first_blocker)
            southeast_ray ^= block_ray

        legal_moves = north_ray | east_ray | south_ray | west_ray | northeast_ray | southeast_ray | southwest_ray | northwest_ray

        # Remove own piece targets
        own_piece_targets = self.get_occupied_squares_by_color[color_to_move]

        if own_piece_targets:
            legal_moves &= ~own_piece_targets

        if color_to_move == Color.WHITE:
            self.white_queen_attacks |= legal_moves

        if color_to_move == Color.BLACK:
            self.black_queen_attacks |= legal_moves

    def update_legal_king_moves(self, move_from_square: np.uint64, color_to_move: bool):
        """
        Pseudo-Legal King Moves: one step in any direction
        :param move_from_square: the proposed square from which the king is to move
        :param color_to_move: the current color to move
        :return: True iff Move is legal
        """
        king_moves = make_uint64()

        # Handle Castling
        # if move.is_castling:
        #     if self.can_castle(self.color_to_move):
        #         king_moves = self.add_castling_moves(king_moves)

        for i in [8, -8]:
            # North-South
            king_moves |= set_bit(king_moves, np.uint64(move_from_square + np.uint64(i)))
        for i in [1, 9, -7]:
            # East (mask the A file)
            king_moves |= set_bit(king_moves, np.uint64(move_from_square + i) & ~np.uint64(File.hexA))
        for i in [-1, -9, 7]:
            # West (mask the H file)
            king_moves |= set_bit(king_moves, np.uint64(move_from_square + i) & ~np.uint64(File.hexH))

        # remove own piece targets
        own_piece_targets = self.get_occupied_squares_by_color[color_to_move]

        if own_piece_targets:
            king_moves &= ~own_piece_targets

        if color_to_move == Color.WHITE:
            self.white_king_attacks |= king_moves

        if color_to_move == Color.BLACK:
            self.black_king_attacks |= king_moves

    @staticmethod
    def add_castling_moves(bitboard: np.uint64, color_to_move: int) -> np.uint64:
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
        castle_ok = [0, 0]

        if not self.castle_rights[color_to_move]:
            return tuple(castle_ok)

        if color_to_move == Color.WHITE:
            kingside_blocked = self.black_attacked_squares() & CastleRoute.WhiteKingside
            queenside_blocked = self.black_attacked_squares() & CastleRoute.WhiteQueenside
            return not kingside_blocked, not queenside_blocked

        if color_to_move == Color.BLACK:
            kingside_blocked = self.white_attacked_squares() & CastleRoute.BlackKingside
            queenside_blocked = self.white_attacked_squares() & CastleRoute.BlackQueenside
            return not kingside_blocked, not queenside_blocked

        return tuple(castle_ok)
