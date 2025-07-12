from wake.constants import Piece

PIECE_VALUES = {
    Piece.wP: 100, Piece.bP: -100,
    Piece.wN: 320, Piece.bN: -320,
    Piece.wB: 330, Piece.bB: -330,
    Piece.wR: 500, Piece.bR: -500,
    Piece.wQ: 900, Piece.bQ: -900,
    Piece.wK: 0, Piece.bK: 0  # King value handled separately
}
