import copy

from wake.bitboard_helpers import get_squares_from_bitboard
from wake.move import Move
from wake.position import Position, evaluate_move_legality


class WakeEngine:
    def __init__(self):
        pass

    @staticmethod
    def generate_rook_moves(pos: Position) -> list:
        moves = []
        rook_attacks = pos.rook_color_map[pos.color_to_move][0]
        rook_piece = pos.rook_color_map[pos.color_to_move][1]
        if not rook_attacks.any():
            return []
        current_rook_locations = pos.piece_map[rook_piece]
        rook_attack_squares = get_squares_from_bitboard(rook_attacks)
        for rook_from_square in list(current_rook_locations):
            for to_square in rook_attack_squares:
                move = Move(rook_piece, (rook_from_square, to_square))
                move = evaluate_move_legality(move, copy.deepcopy(pos))
                if not move.is_illegal_move:
                    moves.append(move)
        return moves

    @staticmethod
    def generate_bishop_moves(pos: Position) -> list:
        moves = []
        bishop_attacks = pos.bishop_color_map[pos.color_to_move][0]
        bishop_piece = pos.bishop_color_map[pos.color_to_move][1]
        if not bishop_attacks.any():
            return []
        current_bishop_locations = list(pos.piece_map[bishop_piece])
        bishop_squares = get_squares_from_bitboard(bishop_attacks)
        for bishop_from_square in current_bishop_locations:
            for to_square in bishop_squares:
                move = Move(bishop_piece, (bishop_from_square, to_square))
                move = evaluate_move_legality(move, copy.deepcopy(pos))
                if not move.is_illegal_move:
                    moves.append(move)
        return moves

    @staticmethod
    def generate_knight_moves(pos: Position) -> list:
        moves = []
        knight_attacks = pos.knight_color_map[pos.color_to_move][0]
        knight_piece = pos.knight_color_map[pos.color_to_move][1]
        if not knight_attacks.any():
            return []
        current_knight_locations = list(pos.piece_map[knight_piece])
        knight_squares = get_squares_from_bitboard(knight_attacks)
        for knight_from_square in current_knight_locations:
            for to_square in knight_squares:
                move = Move(knight_piece, (knight_from_square, to_square))
                move = evaluate_move_legality(move, copy.deepcopy(pos))
                if not move.is_illegal_move:
                    moves.append(move)
        return moves

    @staticmethod
    def generate_king_moves(pos: Position) -> list:
        moves = []
        king_attacks = pos.king_color_map[pos.color_to_move][0] & ~pos.attacked_squares_map[not pos.color_to_move]
        king_piece = pos.king_color_map[pos.color_to_move][1]
        if not king_attacks.any():
            return []
        king_piece_map_copy = pos.piece_map[king_piece].copy()
        king_from_square = king_piece_map_copy.pop()
        king_squares = get_squares_from_bitboard(king_attacks)
        for to_square in king_squares:
            move = Move(king_piece, (king_from_square, to_square))
            move = evaluate_move_legality(move, copy.deepcopy(pos))
            if not move.is_illegal_move:
                moves.append(move)
        return moves

    @staticmethod
    def generate_queen_moves(pos: Position) -> list:
        moves = []
        queen_attacks = pos.queen_color_map[pos.color_to_move][0]
        queen_piece = pos.queen_color_map[pos.color_to_move][1]
        if not queen_attacks.any():
            return []
        current_queen_locations = pos.piece_map[queen_piece]
        queen_squares = get_squares_from_bitboard(queen_attacks)
        for queen_from_square in list(current_queen_locations):
            for to_square in queen_squares:
                move = Move(queen_piece, (queen_from_square, to_square))
                move = evaluate_move_legality(move, copy.deepcopy(pos))
                if not move.is_illegal_move:
                    moves.append(move)
        return moves

    @staticmethod
    def generate_pawn_moves(pos: Position) -> list:
        moves = []
        all_pawn_moves = pos.pawn_color_map[pos.color_to_move][0]
        pawn_piece = pos.pawn_color_map[pos.color_to_move][1]
        if not all_pawn_moves.any():
            return []
        current_pawn_locations = pos.piece_map[pawn_piece]
        pawn_squares = get_squares_from_bitboard(all_pawn_moves)
        for pawn_from_square in list(current_pawn_locations):
            for to_square in pawn_squares:
                move = Move(pawn_piece, (pawn_from_square, to_square))
                move = evaluate_move_legality(move, copy.deepcopy(pos))
                if not move.is_illegal_move:
                    moves.append(move)
        return moves

    def generate_all_legal_moves(self, position: Position):
        moves = []
        moves.extend(self.generate_rook_moves(position))
        moves.extend(self.generate_knight_moves(position))
        moves.extend(self.generate_bishop_moves(position))
        moves.extend(self.generate_king_moves(position))
        moves.extend(self.generate_queen_moves(position))
        moves.extend(self.generate_pawn_moves(position))
        return moves
