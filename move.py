class Move():
    """
    Represents the motion of a piece from an origin square to a target square
    """
    def __init__(self, piece, from_sq, to_sq, is_capture, is_promotion):
        self.piece = piece
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.is_capture = is_capture
        self.is_promotion = is_promotion


    def is_promotion(self):
        # get the promotion type
        pass
