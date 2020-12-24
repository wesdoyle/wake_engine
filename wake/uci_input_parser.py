import re

from wake.constants import Piece, algebraic_square_map
from wake.move import Move


def make_invalid_uci_command():
    command = UciCommand()
    command.is_valid = False
    return command


class UciCommand:
    def __init__(self):
        self.is_valid = False
        self.is_move = False
        self.move = None


async def translate_promote(param):
    promote_piece_map = {
        'q': Piece.QUEEN,
        'n': Piece.KNIGHT,
        'r': Piece.ROOK,
        'b': Piece.BISHOP
    }
    return promote_piece_map.get(param)


class UciInputParser:
    def __init__(self):
        self.move_pattern = re.compile(r'([a-h][1-8])([a-h][1-8])|([qnrb])')

    async def parse_input(self, user_input) -> UciCommand:
        decoded = user_input.decode("utf-8")
        if self.move_pattern.match(decoded):
            match = self.move_pattern.match(decoded)
            from_sq = algebraic_square_map.get(match.groups()[0])
            to_sq = algebraic_square_map.get(match.groups()[1])
            if not from_sq or not to_sq:
                return make_invalid_uci_command()
            move = Move()
            if match.groups()[2]:
                move.is_promotion = True
                move.promote_to = translate_promote(match.groups()[2])
                if not move.promote_to:
                    return make_invalid_uci_command()
            command = UciCommand()
            command.is_valid = True
            command.is_move = True
            command.move = Move(piece=None, squares=(from_sq, to_sq))
            return command
        return make_invalid_uci_command()
