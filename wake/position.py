import numpy as np

from wake.bitboard_helpers import set_bit, get_northwest_ray, bitscan_forward, get_northeast_ray, bitscan_reverse, \
    get_southwest_ray, get_southeast_ray, get_north_ray, get_east_ray, get_south_ray, \
    get_west_ray, make_uint64, pprint_bb, generate_king_attack_bb_from_square
from wake.board import Board
from wake.constants import Color, Piece, Square, CastleRoute, Rank, user_promotion_input, white_promotion_map, \
    black_promotion_map
from wake.move import Move


# TODO: possible side-effects from mutating move all over
#  the place in move legality checking

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

        self.castle_rights = {Color.WHITE: [1, 1], Color.BLACK: [1, 1]}
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

        if move.is_promotion:
            while True:
                self.piece_map[move.piece].remove(move.to_sq)
                promotion_piece = input("Choose promotion piece.")
                promotion_piece = promotion_piece.lower()
                legal_piece = user_promotion_input.get(promotion_piece)
                if not legal_piece:
                    print("Please choose a legal piece")
                    continue
                new_piece = self.get_promotion_piece_type(legal_piece)
                self.piece_map[new_piece].add(move.to_sq)
                break

            self.piece_map[move.piece].add(move.to_sq)

        if move.is_castling:
            self.move_rooks_for_castling(move)

        self.halfmove_clock += 1

        castle_rights = self.castle_rights[self.color_to_move]

        if castle_rights[0] or castle_rights[1]:
            self.adjust_castling_rights(move)

        # Move the pieces
        self.board.update_position_bitboards(self.piece_map)

        # Update the possible moves
        self.update_attack_bitboards()

        self.color_to_move = not self.color_to_move

        return self.generate_fen()

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
        if piece in (Piece.wP, Piece.bP):
            is_legal_pawn_move = self.is_legal_pawn_move(move)
            if not is_legal_pawn_move:
                return False
            if self.is_promotion(move):
                move.is_promotion = True
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

        if move.piece == Piece.wP:
            if not (self.board.white_P_bb & current_square_bb):
                return False
            if self.is_not_pawn_motion_or_attack(move):
                return False
            if (self.white_pawn_attacks & moving_to_square_bb) & ~self.board.black_pieces_bb:
                return False
            return True

        if move.piece == Piece.bP:
            if not (self.board.black_P_bb & current_square_bb):
                return False
            if self.is_not_pawn_motion_or_attack(move):
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
                print("WHITE KING ATTACKS")
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
            kingside_blocked = (self.black_attacked_squares | (self.board.white_pieces_bb & ~self.board.white_K_bb)) & CastleRoute.WhiteKingside
            queenside_blocked = (self.black_attacked_squares | (self.board.white_pieces_bb & ~self.board.white_K_bb)) & CastleRoute.WhiteQueenside
            is_rook_on_h1 = self.board.white_R_bb & set_bit(make_uint64(), Square.H1)
            is_rook_on_a1 = self.board.white_R_bb & set_bit(make_uint64(), Square.A1)
            return [not kingside_blocked.any() and is_rook_on_h1.any(), not queenside_blocked.any() and is_rook_on_a1.any()]

        if color_to_move == Color.BLACK:
            kingside_blocked = (self.white_attacked_squares | (self.board.black_pieces_bb & ~self.board.black_K_bb)) & CastleRoute.BlackKingside
            queenside_blocked = (self.white_attacked_squares | (self.board.black_pieces_bb & ~self.board.black_K_bb)) & CastleRoute.BlackQueenside
            is_rook_on_h8 = self.board.black_R_bb & set_bit(make_uint64(), Square.H8)
            is_rook_on_a8 = self.board.black_R_bb & set_bit(make_uint64(), Square.A8)
            return [not kingside_blocked.any() and is_rook_on_h8.any(), not queenside_blocked.any() and is_rook_on_a8.any()]

    def get_promotion_piece_type(self, legal_piece):
        if self.color_to_move == Color.WHITE:
            return white_promotion_map[legal_piece]
        if self.color_to_move == Color.BLACK:
            return black_promotion_map[legal_piece]

    def generate_fen(self):
        """ TODO: Generate FEN for the current state """
        return '-- TODO: generate FEN --'

