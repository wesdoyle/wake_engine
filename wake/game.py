from wake.constants import Color
from wake.position import Position


class Game:

    def __init__(self):
        self.history = []  # Stack of FENs (TODO: consider stacking position states)
        self.position = Position()
        self.is_over = False
        self.score = [0, 0]

        self.color_to_move = {
            Color.WHITE: "White",
            Color.BLACK: "Black",
        }

    def run(self):
        while not self.is_over:
            move = input(f"{self.color_to_move[self.position.color_to_move]} to move:")
            if not self.try_parse_move(move):
                print("Invalid move format")
                continue
            move_result = self.position.make_move(move)

            if move_result.is_checkmate:
                print("Checkmate")
                self.score[self.position.color_to_move] = 1
                self.is_over = True

            if move_result.is_stalemate:
                print("Stalemate")
                self.score = [0.5, 0.5]
                self.is_over = True

        print(self.score)

    def try_parse_move(self, move):
        pass
