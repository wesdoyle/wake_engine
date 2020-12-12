from position import Position

"""
notes:
    - Game as stack of FENs 
    - each move pushes a position onto the stack
"""


class Game:
    def __init__(self):
        self.history = []  # Stack of Position objects
        initial_position = Position(None)
        self.history.append(initial_position)
        self.half_move_number = 0

    def add_position(self, position) -> bool:
        self.history.append(position)
        self.half_move_number += 1
        return True

    def get_current_position(self) -> Position:
        try:
            return self.history[-1]
        except IndexError:
            print("You've reached the initial position.")

    def get_last_position(self) -> Position:
        try:
            return self.history[-2]
        except IndexError:
            print("You've reached the initial position.")

    def get_move(self, move_number):
        """ TODO: get move by move_number """
        pass

    def take_back_move(self) -> Position:
        try:
            return self.history.pop()
        except IndexError:
            print("You've reached the initial position.")
