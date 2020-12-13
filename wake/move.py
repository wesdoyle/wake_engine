from wake.constants import Piece, Square


class Move:
    """
    Represents the motion of a piece from an origin square to a target square
    """

    def __init__(self, piece, squares):
        self.piece = piece
        self.from_sq = squares[0]
        self.to_sq = squares[1]
        self.is_capture = False
        self.is_promotion = False

    def is_promotion(self):
        # get the promotion type
        pass

    @property
    def is_castling(self):
        if self.piece in Piece.white_pieces:
            return self.from_sq == Square.E1 and self.to_sq in {Square.G1, Square.C1}
        if self.piece in Piece.black_pieces:
            return self.from_sq == Square.E8 and self.to_sq in {Square.G8, Square.C8}
