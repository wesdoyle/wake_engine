import re

from wake.constants import Piece, algebraic_square_map
from wake.move import Move


def make_invalid_uci_command():
    command = UciMove()
    command.is_valid = False
    return command


class UciMove:
    def __init__(self):
        self.is_valid = False
        self.is_move = False
        self.move = None


def translate_promote(param):
    promote_piece_map = {
        'q': Piece.QUEEN,
        'n': Piece.KNIGHT,
        'r': Piece.ROOK,
        'b': Piece.BISHOP
    }
    return promote_piece_map.get(param)


class UciInputParser:
    def __init__(self):
        # Fixed regex: matches moves like "e2e4" or "a7a8q" (with optional promotion)
        self.move_pattern = re.compile(r'^([a-h][1-8])([a-h][1-8])([qnrb])?$')

    def parse_input(self, user_input) -> UciMove:
        user_input = user_input.strip().lower()
        match = self.move_pattern.match(user_input)
        
        if match:
            from_square_str = match.group(1)
            to_square_str = match.group(2)
            promotion_piece = match.group(3)
            
            from_sq = algebraic_square_map.get(from_square_str)
            to_sq = algebraic_square_map.get(to_square_str)
            
            # Handle the case where a1 = 0 (which is falsy but valid)
            if from_sq is None or to_sq is None:
                return make_invalid_uci_command()
                
            move = Move(piece=None, squares=(from_sq, to_sq))
            
            if promotion_piece:
                move.is_promotion = True
                move.promote_to = translate_promote(promotion_piece)
                if not move.promote_to:
                    return make_invalid_uci_command()
            
            command = UciMove()
            command.is_valid = True
            command.is_move = True
            command.move = move
            return command
            
        return make_invalid_uci_command()
