import numpy as np

class Board():

    """
    notes...
        np.array
        [0 , 0, 0,...1,0...0]
        [a1.............h8]

        - function to map fen <-> Board state
        - function to map from geometric <-> bitmap

        ---

        - function to bitwise AND all bitmaps

        -> Piece captures (diff color AND),
            Illegal Moves (same color AND), etc.
    """

    def __init__(self):
        self.white_rook_bitboard = self.create_empty_bitmap()
        self.white_king_bitboard = self.create_empty_bitmap()
        self.white_bishop_bitboard = self.create_empty_bitmap()
        self.white_pawn_bitboard = self.create_empty_bitmap()
        self.white_knight_bitboard = self.create_empty_bitmap()
        self.white_queen_bitboard = self.create_empty_bitmap()

        self.black_rook_bitboard = self.create_empty_bitmap()
        self.black_king_bitboard = self.create_empty_bitmap()
        self.black_bishop_bitboard = self.create_empty_bitmap()
        self.black_pawn_bitboard = self.create_empty_bitmap()
        self.black_knight_bitboard = self.create_empty_bitmap()
        self.black_queen_bitboard = self.create_empty_bitmap()

        self.init_pieces()

        self.bitboards = np.vstack(
                    (self.white_rook_bitboard,
                    self.white_knight_bitboard,
                    self.white_bishop_bitboard,
                    self.white_queen_bitboard,
                    self.white_king_bitboard,
                    self.white_pawn_bitboard,
                    self.black_rook_bitboard,
                    self.black_knight_bitboard,
                    self.black_bishop_bitboard,
                    self.black_queen_bitboard,
                    self.black_king_bitboard,
                    self.black_pawn_bitboard)
                    )

    def update_bitboard_state(self):
        result = np.zeros(64, "byte")
        for board in self.bitboards:
            result = np.bitwise_or(board, result, dtype="byte")
        self.bitboards = result

    def pretty_print_bitboard(self):
        val = ''
        for i, square in enumerate(self.bitboards):
            if not i % 8:
                val += '\n'
            if square:
                val += 'X'
                continue
            val += '-'
        print(val)

    def create_empty_bitmap(self):
        return np.zeros(64, dtype="byte")

    def init_pieces(self):
        self.white_rook_bitboard[0] = 1
        self.white_rook_bitboard[7] = 1
        self.white_knight_bitboard[1] = 1
        self.white_knight_bitboard[6] = 1
        self.white_bishop_bitboard[2] = 1
        self.white_bishop_bitboard[5] = 1
        self.white_queen_bitboard[3] = 1
        self.white_king_bitboard[4] = 1

        for i in range(8, 16):
            self.white_pawn_bitboard[i] = 1

        self.black_rook_bitboard[63] = 1
        self.black_rook_bitboard[56] = 1
        self.black_knight_bitboard[57] = 1
        self.black_knight_bitboard[62] = 1

        self.black_bishop_bitboard[61] = 1
        self.black_bishop_bitboard[58] = 1

        self.black_queen_bitboard[59] = 1
        self.black_king_bitboard[60] = 1

        for i in range(48, 56):
            self.black_pawn_bitboard[i] = 1
