import threading
from wake.bitboard_helpers import pprint_pieces
from wake.constants import Color
from wake.position import Position
from wake.uci_input_parser import UciInputParser


class Game:

    def __init__(self):
        self.history = []  # Stack of FENs (TODO: consider stacking position states)
        self.position = Position()
        self.is_over = False
        self.score = [0, 0]
        self.parser = UciInputParser()

        self.color_to_move = {
            Color.WHITE: "White",
            Color.BLACK: "Black",
        }

    def run(self):
        choose_mode = input("Choose input mode (UCI is the only mode available): ")

        if choose_mode.strip().lower() == 'uci':
            self.run_uci_mode()

        else:
            self.run()

        print(self.score)

    def try_parse_move(self, move):
        engine_input = self.parser.parse_input(move)
        if not engine_input.is_valid:
            print("Invalid input")
            return None
        if engine_input.is_move:
            move_piece = self.position.get_piece_on_square(engine_input.move.from_sq)
            if not move_piece:
                print("Invalid input")
                return None
            engine_input.move.piece = move_piece
            return engine_input.move

    def run_uci_mode(self):
        while not self.is_over:
            uci_input = input(f"{self.color_to_move[self.position.color_to_move]} to move:")

            move = self.try_parse_move(uci_input)

            if not move:
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

            self.history.append(move_result.fen)
            pprint_pieces(self.position.piece_map)


if __name__ == "__main__":
    game = Game()
    game.run()
