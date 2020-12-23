import re

from wake.move import Move


class UciInputParser:
    def __init__(self):
        self.input_pattern = re.compile(r'([a-h][1-8])([a-h][1-8])|([qnrb])')

    def parse_input(self, user_input) -> bool or Move:
        if not self.input_pattern.match(user_input):
            return False