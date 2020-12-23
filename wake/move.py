from wake.constants import Piece


class Move:
    """
    Represents the motion of a piece from an origin square to a target square
    """

    def __init__(self, piece, squares):
        self.piece = piece
        self.from_sq = squares[0]
        self.to_sq = squares[1]
        self.is_capture = False
        self.is_en_passant = False
        self.is_promotion = False
        self.is_castling = False

    @property
    def color(self):
        if self.piece in Piece.white_pieces:
            return 0
        return 1


class MoveResult:
    """
    Represents the positional outcome of a move
    """
    def __init__(self):
        self.is_checkmate = False
        self.is_king_in_check = False
        self.is_stalemate = False
        self.is_draw_claim_allowed = False
        self.is_illegal_move = False
        self.fen = ''
