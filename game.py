"""
notes:
    - Game as stack of Positions
    - each move pushes a position onto the stack
"""
class Game:
    def __init__(self):
        self.history = []
