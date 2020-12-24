import re

from wake.move import Move


def make_invalid_uci_command():
    command = UciCommand()
    command.is_valid = False


class UciInputParser:
    def __init__(self):
        self.move_pattern = re.compile(r'([a-h][1-8])([a-h][1-8])|([qnrb])')

    def parse_input(self, user_input) -> bool or Move:
        if self.move_pattern.match(user_input):
            match = self.move_pattern.match(user_input)
            from_sq = match[0]
            to_sq = match[1]
            move = Move()
            if match[2]:
                move.is_promotion = True
            command = UciCommand()
            command.is_valid = True
            command.is_move = True
            command.move = Move(piece=None, squares=(from_sq, to_sq))
            return command
        return make_invalid_uci_command()


class UciCommand:
    def __init__(self):
        self.is_valid = False
        self.is_move = False
        self.move = None
