import copy

import numpy as np

from wake.bitboard_helpers import set_bit, get_northwest_ray, bitscan_forward, get_northeast_ray, bitscan_reverse, \
    get_southwest_ray, get_southeast_ray, get_north_ray, get_east_ray, get_south_ray, \
    get_west_ray, make_uint64, pprint_bb, generate_king_attack_bb_from_square
from wake.board import Board
from wake.constants import Color, Piece, Square, CastleRoute, Rank, user_promotion_input, white_promotion_map, \
    black_promotion_map, square_map
from wake.move import Move


# TODO: possible side-effects from mutating move all over
#  the place in move legality checking

# position reached 3-times draw allowed

# implement 50 move rule

# Use existing code / de-dupe from board module

# Unit tests

# King in check / Checkmate code

# Push FEN to the Game stack

# Should we generate the move from the board?

class PositionState:
    """
    Memento for Position
    """

    def __init__(self, kwargs):
        self.board = kwargs['board']
        self.color_to_move = kwargs['color_to_move']
        self.castle_rights = kwargs['castle_rights']
        self.en_passant_side = kwargs['en_passant_side']
        self.en_passant_target = kwargs['en_passant_target']
        self.is_en_passant_capture = kwargs['is_en_passant_capture']
        self.halfmove_clock = kwargs['halfmove_clock']
        self.halfmove = kwargs['halfmove']
        self.king_in_check = kwargs['king_in_check']
        self.piece_map = kwargs['piece_map']
        self.white_pawn_moves = kwargs['white_pawn_moves']
        self.white_pawn_attacks = kwargs['white_pawn_attacks']
        self.white_rook_attacks = kwargs['white_rook_attacks']
        self.white_knight_attacks = kwargs['white_knight_attacks']
        self.white_bishop_attacks = kwargs['white_bishop_attacks']
        self.white_queen_attacks = kwargs['white_queen_attacks']
        self.white_king_attacks = kwargs['white_king_attacks']
        self.black_pawn_moves = kwargs['black_pawn_moves']
        self.black_pawn_attacks = kwargs['black_pawn_attacks']
        self.black_rook_attacks = kwargs['black_rook_attacks']
        self.black_knight_attacks = kwargs['black_knight_attacks']
        self.black_bishop_attacks = kwargs['black_bishop_attacks']
        self.black_queen_attacks = kwargs['black_queen_attacks']
        self.black_king_attacks = kwargs['black_king_attacks']


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

        # [kingside, queenside] castle rights
        self.castle_rights = {Color.WHITE: [1, 1], Color.BLACK: [1, 1]}

        self.en_passant_target = None  # target square
        self.en_passant_side = Color.WHITE
        self.is_en_passant_capture = False
        self.halfmove_clock = 0
        self.halfmove = 2

        # [white, black] boolean king is in check
        self.king_in_check = [0, 0]

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

    def reset_state_to(self, memento: PositionState) -> None:
        for k, v in memento.__dict__.items():
            setattr(self, k, v)

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

        if self.king_in_check[self.color_to_move] and not self.any_legal_moves():
            pass # CHECKMATE

        original_position = PositionState(copy.deepcopy(self.__dict__))

        if not self.is_legal_move(move):
            print("Illegal move")
            return

        if move.is_capture:
            self.halfmove_clock = 0
            self.remove_opponent_piece_from_square(move.to_sq)

        if self.is_en_passant_capture:
            if self.color_to_move == Color.WHITE:
                self.remove_opponent_piece_from_square(move.to_sq - 8)
            if self.color_to_move == Color.BLACK:
                self.remove_opponent_piece_from_square(move.to_sq + 8)

        self.is_en_passant_capture = False

        if move.piece in {Piece.wP, Piece.bP}:
            self.halfmove_clock = 0
        self.piece_map[move.piece].remove(move.from_sq)
        self.piece_map[move.piece].add(move.to_sq)

        if move.is_promotion:
            while True:
                promotion_piece = input("Choose promotion piece.")
                promotion_piece = promotion_piece.lower()
                legal_piece = user_promotion_input.get(promotion_piece)
                if not legal_piece:
                    print("Please choose a legal piece")
                    continue
                self.piece_map[move.piece].remove(move.to_sq)
                new_piece = self.get_promotion_piece_type(legal_piece)
                self.piece_map[new_piece].add(move.to_sq)
                break

        if move.is_castling:
            self.move_rooks_for_castling(move)

        self.halfmove_clock += 1
        self.halfmove += 1

        castle_rights = self.castle_rights[self.color_to_move]

        if castle_rights[0] or castle_rights[1]:
            self.adjust_castling_rights(move)

        if self.en_passant_side != self.color_to_move:
            self.en_passant_target = None

        # Move the pieces
        self.board.update_position_bitboards(self.piece_map)

        # Update the possible moves
        self.update_attack_bitboards()

        self.evaluate_king_check()

        if self.king_in_check[self.color_to_move]:
            self.reset_state_to(original_position)
            print("Illegal move: own king in check")
            return self.generate_fen()

        self.color_to_move = not self.color_to_move

        # Commit the bitboards
        return self.generate_fen()

    def remove_opponent_piece_from_square(self, to_sq):
        # TODO: Inefficient
        target = None
        for k, v in self.piece_map.items():
            if to_sq in v:
                target = k
                break
        self.piece_map[target].remove(to_sq)

    def update_attack_bitboards(self):
        self.reset_attack_bitboards()
        for piece, squares in self.piece_map.items():

            # PAWNS
            if piece == Piece.wP:
                for square in squares:
                    self.update_legal_pawn_moves(square, Color.WHITE)
            if piece == Piece.bP:
                for square in squares:
                    self.update_legal_pawn_moves(square, Color.BLACK)

            # ROOKS
            if piece == Piece.wR:
                for square in squares:
                    self.update_legal_rook_moves(square, Color.WHITE)
            if piece == Piece.bR:
                for square in squares:
                    self.update_legal_rook_moves(square, Color.BLACK)

            # KNIGHTS
            if piece == Piece.wN:
                for square in squares:
                    self.update_legal_knight_moves(square, Color.WHITE)
            if piece == Piece.bN:
                for square in squares:
                    self.update_legal_knight_moves(square, Color.BLACK)

            # BISHOPS
            if piece == Piece.wB:
                for square in squares:
                    self.update_legal_bishop_moves(square, Color.WHITE)
            if piece == Piece.bB:
                for square in squares:
                    self.update_legal_bishop_moves(square, Color.BLACK)

            # QUEENS
            if piece == Piece.wQ:
                for square in squares:
                    self.update_legal_queen_moves(square, Color.WHITE)
            if piece == Piece.bQ:
                for square in squares:
                    self.update_legal_queen_moves(square, Color.BLACK)

            # KINGS
            if piece == Piece.wK:
                for square in squares:
                    self.update_legal_king_moves(square, Color.WHITE)

            if piece == Piece.bK:
                for square in squares:
                    self.update_legal_king_moves(square, Color.BLACK)

    def adjust_castling_rights(self, move):
        if move.piece in {Piece.wK, Piece.bK, Piece.wR, Piece.bR}:
            if move.piece == Piece.wK:
                self.castle_rights[Color.WHITE] = [0, 0]
            if move.piece == Piece.bK:
                self.castle_rights[Color.BLACK] = [0, 0]

            if move.piece == Piece.wR:
                if move.from_sq == Square.H1:
                    self.castle_rights[Color.WHITE][0] = 0
                if move.from_sq == Square.A1:
                    self.castle_rights[Color.WHITE][1] = 0
            if move.piece == Piece.bR:
                if move.from_sq == Square.H8:
                    self.castle_rights[Color.WHITE][0] = 0
                if move.from_sq == Square.A8:
                    self.castle_rights[Color.WHITE][1] = 0

    def move_rooks_for_castling(self, move):
        rook_color_map = {
            Color.WHITE: Piece.wR,
            Color.BLACK: Piece.bR
        }
        square_map = {
            Square.G1: (Square.H1, Square.F1),
            Square.C1: (Square.A1, Square.D1),
            Square.G8: (Square.H8, Square.F8),
            Square.C8: (Square.A8, Square.D8),
        }
        self.piece_map[rook_color_map[self.color_to_move]].remove(square_map[move.to_sq][0])
        self.piece_map[rook_color_map[self.color_to_move]].add(square_map[move.to_sq][1])

    def reset_attack_bitboards(self):
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

    # -------------------------------------------------------------
    # MOVE LEGALITY CHECKING
    # -------------------------------------------------------------

    def is_legal_move(self, move: Move) -> bool:
        """
        For a given move, returns True iff it is legal given the Position state
        """
        piece = move.piece

        if self.is_capture(move):
            move.is_capture = True

        if piece in (Piece.wP, Piece.bP):
            is_legal_pawn_move = self.is_legal_pawn_move(move)

            if not is_legal_pawn_move:
                return False

            if self.is_promotion(move):
                move.is_promotion = True
                return True

            en_passant_target = self.try_get_en_passant_target(move)

            if en_passant_target:
                self.en_passant_side = self.color_to_move
                self.en_passant_target = int(en_passant_target)

            if move.to_sq == self.en_passant_target:
                self.is_en_passant_capture = True

            return True

        if piece in (Piece.wB, Piece.bB):
            return self.is_legal_bishop_move(move)

        if piece in (Piece.wR, Piece.bR):
            return self.is_legal_rook_move()

        if piece in (Piece.wN, Piece.bN):
            return self.is_legal_knight_move(move)

        if piece in (Piece.wQ, Piece.bQ):
            return self.is_legal_queen_move(move)

        if piece in (Piece.wK, Piece.bK):
            is_legal_king_move = self.is_legal_king_move(move)
            if not is_legal_king_move:
                return False
            if self.is_castling(move):
                move.is_castling = True
            return True

    def try_get_en_passant_target(self, move):
        if move.piece not in {Piece.wP, Piece.bP}:
            return None
        if self.color_to_move == Color.WHITE:
            if move.to_sq in Rank.x4 and move.from_sq in Rank.x2:
                return move.to_sq - 8
        if self.color_to_move == Color.BLACK:
            if move.to_sq in Rank.x5 and move.from_sq in Rank.x7:
                return move.to_sq + 8

    def is_capture(self, move):
        moving_to_square_bb = set_bit(make_uint64(), move.to_sq)

        if self.color_to_move == Color.WHITE:
            intersects = moving_to_square_bb & self.board.black_pieces_bb
            if intersects.any():
                return True
        if self.color_to_move == Color.BLACK:
            intersects = moving_to_square_bb & self.board.white_pieces_bb
            if intersects.any():
                return True
        return False

    # -------------------------------------------------------------
    # PIECE MOVE LEGALITY BY PIECE
    # -------------------------------------------------------------

    def is_legal_pawn_move(self, move: Move) -> bool:
        """
        Returns True iff the given pawn move is legal - i.e.
        - the to square intersects with pawn "motion" (forward) bitboard
        - the to square is an attack and intersects with opponent piece or en passant target bitboard
        """
        current_square_bb = set_bit(make_uint64(), move.from_sq)
        moving_to_square_bb = set_bit(make_uint64(), move.to_sq)
        en_passant_target = make_uint64()

        if self.en_passant_target:
            en_passant_target = set_bit(make_uint64(), self.en_passant_target)

        if move.piece == Piece.wP:
            if not (self.board.white_P_bb & current_square_bb):
                return False
            if self.is_not_pawn_motion_or_attack(move):
                return False

            # If it's a pawn attack move, check that it intersects with black pieces or ep target
            if move.from_sq == move.to_sq - 9 or move.from_sq == move.to_sq - 7:
                if (self.white_pawn_attacks & moving_to_square_bb) & ~(self.board.black_pieces_bb | en_passant_target):
                    pprint_bb(en_passant_target)
                    return False
            return True

        if move.piece == Piece.bP:
            if not (self.board.black_P_bb & current_square_bb):
                return False
            if self.is_not_pawn_motion_or_attack(move):
                return False

            # If it's a pawn attack move, check that it intersects with white pieces or ep target
            if move.from_sq == move.to_sq + 9 or move.from_sq == move.to_sq + 7:
                if (self.black_pawn_attacks & moving_to_square_bb) & ~(self.board.white_pieces_bb | en_passant_target):
                    return False
            return True

    def is_legal_bishop_move(self, move: Move) -> bool:
        """
        Returns True iff the given bishop move is legal - i.e.
        - the to move intersects with the bishop attack bitboard
        """
        current_square_bb = set_bit(make_uint64(), move.from_sq)
        if move.piece == Piece.wB:
            if not (self.board.white_B_bb & current_square_bb):
                return False
            if self.is_not_bishop_attack(move.to_sq):
                return False
            return True
        if move.piece == Piece.bB:
            if not (self.board.black_B_bb & current_square_bb):
                return False
            if self.is_not_bishop_attack(move.to_sq):
                return False
            return True

    def is_legal_rook_move(self, move: Move) -> bool:
        current_square_bb = set_bit(make_uint64(), move.from_sq)
        if move.piece == Piece.wR:
            if not (self.board.white_R_bb & current_square_bb):
                return False
            if self.is_not_rook_attack(move.to_sq):
                return False
            return True
        if move.piece == Piece.bR:
            if not (self.board.black_R_bb & current_square_bb):
                return False
            if self.is_not_rook_attack(move.to_sq):
                return False
            return True

    def is_legal_knight_move(self, move):
        current_square_bb = set_bit(make_uint64(), move.from_sq)
        if move.piece == Piece.wN:
            if not (self.board.white_N_bb & current_square_bb):
                return False
            if self.is_not_knight_attack(move.to_sq):
                return False
            return True
        if move.piece == Piece.bN:
            if not (self.board.black_N_bb & current_square_bb):
                return False
            if self.is_not_knight_attack(move.to_sq):
                return False
            return True

    def is_legal_queen_move(self, move):
        current_square_bb = set_bit(make_uint64(), move.from_sq)
        if move.piece == Piece.wQ:
            if not (self.board.white_Q_bb & current_square_bb):
                return False
            if self.is_not_queen_attack(move.to_sq):
                return False
            return True
        if move.piece == Piece.bQ:
            if not (self.board.black_Q_bb & current_square_bb):
                return False
            if self.is_not_queen_attack(move.to_sq):
                return False
            return True

    def is_legal_king_move(self, move):
        current_square_bb = set_bit(make_uint64(), move.from_sq)
        if move.piece == Piece.wK:
            if not (self.board.white_K_bb & current_square_bb):
                return False
            if self.is_not_king_attack(move.to_sq):
                return False
            return True
        if move.piece == Piece.bK:
            if not (self.board.black_K_bb & current_square_bb):
                return False
            if self.is_not_king_attack(move.to_sq):
                return False
            return True

    def is_wrong_color_piece(self, move):
        if self.color_to_move == Color.WHITE and move.piece not in Piece.white_pieces:
            return False

        if self.color_to_move == Color.BLACK and move.piece not in Piece.black_pieces:
            return False

    # -------------------------------------------------------------
    # PIECE MOVE LEGALITY CHECKING HELPERS
    # -------------------------------------------------------------

    def is_not_pawn_motion_or_attack(self, move):
        to_sq_bb = set_bit(make_uint64(), move.to_sq)
        if self.color_to_move == Color.WHITE:
            if not (self.board.white_pawn_motion_bbs[move.from_sq]
                    | self.board.white_pawn_attack_bbs[move.from_sq]) & to_sq_bb:
                return True
        if self.color_to_move == Color.BLACK:
            if not (self.board.black_pawn_motion_bbs[move.from_sq]
                    | self.board.black_pawn_attack_bbs[move.from_sq]) & to_sq_bb:
                return True

    def is_not_bishop_attack(self, to_sq):
        moving_to_square_bb = set_bit(make_uint64(), to_sq)
        if self.color_to_move == Color.WHITE:
            if not (self.white_bishop_attacks & moving_to_square_bb):
                return True
        if self.color_to_move == Color.BLACK:
            if not (self.black_bishop_attacks & moving_to_square_bb):
                return True

    def is_not_knight_attack(self, to_sq):
        moving_to_square_bb = set_bit(make_uint64(), to_sq)
        if self.color_to_move == Color.WHITE:
            if not (self.white_knight_attacks & moving_to_square_bb):
                return True
        if self.color_to_move == Color.BLACK:
            if not (self.black_knight_attacks & moving_to_square_bb):
                return True

    def is_not_king_attack(self, to_sq):
        moving_to_square_bb = set_bit(make_uint64(), to_sq)
        if self.color_to_move == Color.WHITE:
            if not (self.white_king_attacks & moving_to_square_bb):
                pprint_bb(self.white_king_attacks)
                return True
        if self.color_to_move == Color.BLACK:
            if not (self.black_king_attacks & moving_to_square_bb):
                return True
        return False

    def is_not_queen_attack(self, to_sq):
        moving_to_square_bb = set_bit(make_uint64(), to_sq)
        if self.color_to_move == Color.WHITE:
            if not (self.white_queen_attacks & moving_to_square_bb):
                return True
        if self.color_to_move == Color.BLACK:
            if not (self.black_queen_attacks & moving_to_square_bb):
                return True

    def is_not_rook_attack(self, to_sq):
        moving_to_square_bb = set_bit(make_uint64(), to_sq)
        if self.color_to_move == Color.WHITE:
            if not (self.white_rook_attacks & moving_to_square_bb):
                return False
        if self.color_to_move == Color.BLACK:
            if not (self.black_rook_attacks & moving_to_square_bb):
                return False

    @staticmethod
    def is_castling(move):
        if move.from_sq == Square.E1:
            if move.to_sq in {Square.G1, Square.C1}:
                return True
        if move.from_sq == Square.E8:
            if move.to_sq in {Square.G8, Square.E8}:
                return True
        return False

    def is_promotion(self, pawn_move):
        if self.color_to_move == Color.WHITE and pawn_move.to_sq in Rank.x8:
            return True
        if self.color_to_move == Color.BLACK and pawn_move.to_sq in Rank.x1:
            return True
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

    # -------------------------------------------------------------
    # LEGAL PAWN MOVES
    # -------------------------------------------------------------

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

        self.white_pawn_attacks = self.board.white_pawn_attacks
        self.black_pawn_attacks = self.board.black_pawn_attacks

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
            self.white_pawn_attacks |= legal_attack_moves[Color.WHITE]
            self.white_pawn_moves |= legal_non_attack_moves[Color.WHITE]

        if color_to_move == Color.BLACK:
            self.black_pawn_attacks |= legal_attack_moves[Color.BLACK]
            self.black_pawn_moves |= legal_non_attack_moves[Color.BLACK]

    # -------------------------------------------------------------
    # LEGAL QUEEN MOVES
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # LEGAL KING MOVES
    # -------------------------------------------------------------

    def update_legal_king_moves(self, move_from_square: int, color_to_move: bool):
        """
        Pseudo-Legal King Moves: one step in any direction
        :param move_from_square: the proposed square from which the king is to move
        :param color_to_move: the current color to move
        :return: True iff Move is legal
        """

        king_moves = generate_king_attack_bb_from_square(move_from_square)

        own_piece_targets = None

        if color_to_move == Color.WHITE:
            own_piece_targets = self.board.white_pieces_bb

        if color_to_move == Color.BLACK:
            own_piece_targets = self.board.black_pieces_bb

        if own_piece_targets.any():
            king_moves &= ~own_piece_targets

        # Handle Castling
        can_castle = self.can_castle(color_to_move)

        if can_castle[0] or can_castle[1]:
            king_moves |= self.add_castling_moves(king_moves, can_castle, color_to_move)

        if color_to_move == Color.WHITE:
            self.white_king_attacks |= king_moves

        if color_to_move == Color.BLACK:
            self.black_king_attacks |= king_moves

    @staticmethod
    def add_castling_moves(bitboard: np.uint64, can_castle: list, color_to_move) -> np.uint64:
        """
        Adds castling squares to the bitboard
        :param bitboard: numpy uint64 bitboard
        :return:
        """
        if color_to_move == Color.WHITE:
            if can_castle[0]:
                bitboard |= set_bit(bitboard, Square.G1)
            if can_castle[1]:
                bitboard |= set_bit(bitboard, Square.C1)

        if color_to_move == Color.BLACK:
            if can_castle[0]:
                bitboard |= set_bit(bitboard, Square.G8)
            if can_castle[1]:
                bitboard |= set_bit(bitboard, Square.C8)

        return bitboard

    def can_castle(self, color_to_move) -> list:
        """
        Returns a tuple of (bool, bool) can castle on (kingside, queenside)
        and can move through the castling squares without being in check.
        :return: Tuple (bool, bool) iff the color_to_move has castling rights
        and can move through the castling squares without being in check on (kingside, queenside)
        """
        castle_rights = self.castle_rights[color_to_move]

        if not castle_rights[0] or not castle_rights[1]:
            return [0, 0]

        if color_to_move == Color.WHITE:
            kingside_blocked = (self.black_attacked_squares | (
                    self.board.white_pieces_bb & ~self.board.white_K_bb)) & CastleRoute.WhiteKingside
            queenside_blocked = (self.black_attacked_squares | (
                    self.board.white_pieces_bb & ~self.board.white_K_bb)) & CastleRoute.WhiteQueenside
            is_rook_on_h1 = self.board.white_R_bb & set_bit(make_uint64(), Square.H1)
            is_rook_on_a1 = self.board.white_R_bb & set_bit(make_uint64(), Square.A1)
            return [not kingside_blocked.any() and is_rook_on_h1.any(),
                    not queenside_blocked.any() and is_rook_on_a1.any()]

        if color_to_move == Color.BLACK:
            kingside_blocked = (self.white_attacked_squares | (
                    self.board.black_pieces_bb & ~self.board.black_K_bb)) & CastleRoute.BlackKingside
            queenside_blocked = (self.white_attacked_squares | (
                    self.board.black_pieces_bb & ~self.board.black_K_bb)) & CastleRoute.BlackQueenside
            is_rook_on_h8 = self.board.black_R_bb & set_bit(make_uint64(), Square.H8)
            is_rook_on_a8 = self.board.black_R_bb & set_bit(make_uint64(), Square.A8)
            return [not kingside_blocked.any() and is_rook_on_h8.any(),
                    not queenside_blocked.any() and is_rook_on_a8.any()]

    def get_promotion_piece_type(self, legal_piece):
        if self.color_to_move == Color.WHITE:
            return white_promotion_map[legal_piece]
        if self.color_to_move == Color.BLACK:
            return black_promotion_map[legal_piece]

    def generate_fen(self) -> str:
        black_pawn_bin = np.binary_repr(self.board.black_P_bb, 64)
        black_rook_bin = np.binary_repr(self.board.black_R_bb, 64)
        black_knight_bin = np.binary_repr(self.board.black_N_bb, 64)
        black_bishop_bin = np.binary_repr(self.board.black_B_bb, 64)
        black_queen_bin = np.binary_repr(self.board.black_Q_bb, 64)
        black_king_bin = np.binary_repr(self.board.black_K_bb, 64)
        white_pawn_bin = np.binary_repr(self.board.white_P_bb, 64)
        white_rook_bin = np.binary_repr(self.board.white_R_bb, 64)
        white_knight_bin = np.binary_repr(self.board.white_N_bb, 64)
        white_bishop_bin = np.binary_repr(self.board.white_B_bb, 64)
        white_queen_bin = np.binary_repr(self.board.white_Q_bb, 64)
        white_king_bin = np.binary_repr(self.board.white_K_bb, 64)

        black_pawn_squares = [i for i in range(len(black_pawn_bin)) if int(black_pawn_bin[i])]
        black_rook_squares = [i for i in range(len(black_rook_bin)) if int(black_rook_bin[i])]
        black_knight_squares = [i for i in range(len(black_knight_bin)) if int(black_knight_bin[i])]
        black_bishop_squares = [i for i in range(len(black_bishop_bin)) if int(black_bishop_bin[i])]
        black_queen_squares = [i for i in range(len(black_queen_bin)) if int(black_queen_bin[i])]
        black_king_square = [i for i in range(len(black_king_bin)) if int(black_king_bin[i])]
        white_pawn_squares = [i for i in range(len(white_pawn_bin)) if int(white_pawn_bin[i])]
        white_rook_squares = [i for i in range(len(white_rook_bin)) if int(white_rook_bin[i])]
        white_knight_squares = [i for i in range(len(white_knight_bin)) if int(white_knight_bin[i])]
        white_bishop_squares = [i for i in range(len(white_bishop_bin)) if int(white_bishop_bin[i])]
        white_queen_squares = [i for i in range(len(white_queen_bin)) if int(white_queen_bin[i])]
        white_king_square = [i for i in range(len(white_king_bin)) if int(white_king_bin[i])]

        fen_dict = {i: None for i in range(64)}

        for s in black_pawn_squares:
            fen_dict[s] = 'p'
        for s in black_rook_squares:
            fen_dict[s] = 'r'
        for s in black_knight_squares:
            fen_dict[s] = 'n'
        for s in black_bishop_squares:
            fen_dict[s] = 'b'
        for s in black_queen_squares:
            fen_dict[s] = 'q'
        for s in black_king_square:
            fen_dict[s] = 'k'
        for s in white_pawn_squares:
            fen_dict[s] = 'P'
        for s in white_rook_squares:
            fen_dict[s] = 'R'
        for s in white_knight_squares:
            fen_dict[s] = 'N'
        for s in white_bishop_squares:
            fen_dict[s] = 'B'
        for s in white_queen_squares:
            fen_dict[s] = 'Q'
        for s in white_king_square:
            fen_dict[s] = 'K'

        fen = ''
        empty = 0
        row = ''

        for k, v in fen_dict.items():
            if not k % 8 and not k == 0:
                if empty:
                    row += str(empty)
                    empty = 0
                fen += row[::-1] + "/"
                row = ''

            if not v:
                empty += 1
                continue

            if empty:
                row += str(empty)
                empty = 0

            row += v
        fen += row[::-1]

        side_to_move_map = {
            Color.WHITE: 'w',
            Color.BLACK: 'b'
        }

        fen += f" {side_to_move_map[self.color_to_move]}"

        w_castle_king = self.castle_rights[Color.WHITE][0] == True
        w_castle_queen = self.castle_rights[Color.WHITE][1] == True
        b_castle_king = self.castle_rights[Color.BLACK][0] == True
        b_castle_queen = self.castle_rights[Color.BLACK][1] == True

        fen += " "

        if w_castle_king:
            fen += "K"
        if w_castle_queen:
            fen += "Q"
        if b_castle_king:
            fen += "k"
        if b_castle_queen:
            fen += "q"

        if not w_castle_king and not w_castle_queen and not b_castle_king and not b_castle_queen:
            fen += "-"

        fen += " "

        if self.en_passant_target:
            fen += str(square_map[self.en_passant_target])
        else:
            fen += "-"

        # Halfmove clock
        fen += f" {str(self.halfmove_clock)}"

        # Full-move Number
        fen += f" {str(self.halfmove // 2)}"

        return fen

    def evaluate_king_check(self):
        """
        Evaluates instance state for intersection of attacked squares and opposing king position.
        Updates instance state `king_in_check` for the corresponding color
        """
        if self.black_attacked_squares & self.board.white_K_bb:
            self.king_in_check[0] = 1
        else:
            self.king_in_check[0] = 0
        if self.white_attacked_squares & self.board.black_K_bb:
            self.king_in_check[1] = 1
        else:
            self.king_in_check[1] = 0
