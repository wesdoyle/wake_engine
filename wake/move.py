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
