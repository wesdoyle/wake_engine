import numpy as np

class Board():

    """
    Scratch notes
        - square-centric representation
        - function to map fen <-> Board state
        - function to map from geometric <-> bitmap
        ---

        - function to bitwise AND all bitmaps

        -> Piece captures (diff color AND),
            Illegal Moves (same color AND), etc.
    """

    def __init__(self):
        self.white_R_bb = self.create_empty_bitmap()
        self.white_K_bb = self.create_empty_bitmap()
        self.white_B_bb = self.create_empty_bitmap()
        self.white_P_bb = self.create_empty_bitmap()
        self.white_N_bb = self.create_empty_bitmap()
        self.white_Q_bb = self.create_empty_bitmap()

        self.black_R_bb = self.create_empty_bitmap()
        self.black_K_bb = self.create_empty_bitmap()
        self.black_B_bb = self.create_empty_bitmap()
        self.black_P_bb = self.create_empty_bitmap()
        self.black_N_bb = self.create_empty_bitmap()
        self.black_Q_bb = self.create_empty_bitmap()

        self.init_pieces()

        self.occupied_squares_bb = np.vstack(
                    (self.white_R_bb,
                    self.white_N_bb,
                    self.white_B_bb,
                    self.white_Q_bb,
                    self.white_K_bb,
                    self.white_P_bb,
                    self.black_R_bb,
                    self.black_N_bb,
                    self.black_B_bb,
                    self.black_Q_bb,
                    self.black_K_bb,
                    self.black_P_bb)
                    )

    def init_pieces(self):
        self._set_white()
        self._set_black()

    def update_occupied_squares_bb(self):
        result = np.zeros(64, "byte")
        for board in self.occupied_squares_bb:
            result = np.bitwise_or(board, result, dtype="byte")
        self.occupied_squares_bb = result

    def get_empty_squares_bb(self):
        return  1 - self.occupied_squares_bb

    def create_empty_bitmap(self):
        return np.zeros(64, dtype="byte")

    def _set_white(self):
        for i in range(8, 16):
            self.white_P_bb[i] = 1
        self.white_R_bb[0] = 1
        self.white_R_bb[7] = 1
        self.white_N_bb[1] = 1
        self.white_N_bb[6] = 1
        self.white_B_bb[2] = 1
        self.white_B_bb[5] = 1
        self.white_Q_bb[3] = 1
        self.white_K_bb[4] = 1

    def _set_black(self):
        for i in range(48, 56):
            self.black_P_bb[i] = 1
        self.black_R_bb[63] = 1
        self.black_R_bb[56] = 1
        self.black_N_bb[57] = 1
        self.black_N_bb[62] = 1
        self.black_B_bb[61] = 1
        self.black_B_bb[58] = 1
        self.black_Q_bb[59] = 1
        self.black_K_bb[60] = 1

def pretty_print_bb(bb):
    val = ''
    display_rank = 8
    board = np.reshape(np.flip(bb), (8, 8))
    for i, row in enumerate(board):
        val += f'{display_rank} '
        display_rank -= 1
        for square in np.flip(row):
            if square:
                val += ' ▓'
                continue
            val += ' ░'
        val += '\n'
    val += '  '
    for char in 'ABCDEFGH':
        val += f' {char}'
    print(val)
