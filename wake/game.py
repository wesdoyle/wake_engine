from wake.constants import Color, Piece
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
        "p e2 e4"
        if move.lower().strip() not in console_piece_notation_map:
            return False


console_piece_notation_map = {
    "wp": Piece.wP,
    "wb": Piece.wB,
    "wn": Piece.wN,
    "wk": Piece.wK,
    "wr": Piece.wR,
    "wq": Piece.wQ,

    "bp": Piece.bP,
    "bb": Piece.bB,
    "bn": Piece.bN,
    "bk": Piece.bK,
    "br": Piece.bR,
    "bq": Piece.bQ,
}
