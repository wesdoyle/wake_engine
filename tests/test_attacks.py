"""
Tests for the bitboard attacks module.

Tests all attack pattern generation functions for different piece types
including knights, rooks, bishops, queens, kings, and pawns.
"""

import numpy as np

from wake.bitboard.attacks import (
    generate_knight_attack_bb_from_square,
    generate_rank_attack_bb_from_square,
    generate_file_attack_bb_from_square,
    generate_rook_attack_bb_from_square,
    generate_diag_attack_bb_from_square,
    generate_queen_attack_bb_from_square,
    generate_king_attack_bb_from_square,
    generate_white_pawn_attack_bb_from_square,
    generate_black_pawn_attack_bb_from_square,
    generate_white_pawn_motion_bb_from_square,
    generate_black_pawn_motion_bb_from_square,
)
from wake.bitboard.core import get_squares_from_bitboard


class TestKnightAttacks:
    """Test knight attack pattern generation."""

    def test_knight_attack_from_center(self):
        """Test knight attacks from center square."""
        attacks = generate_knight_attack_bb_from_square(28)  # e4
        squares = get_squares_from_bitboard(attacks)

        # Knight from e4 can attack: d2, f2, c3, g3, c5, g5, d6, f6
        expected = [11, 13, 18, 22, 34, 38, 43, 45]
        assert sorted(squares) == sorted(expected)

    def test_knight_attack_from_corner(self):
        """Test knight attacks from corner square."""
        attacks = generate_knight_attack_bb_from_square(0)  # a1
        squares = get_squares_from_bitboard(attacks)

        # Knight from a1 can only attack b3 and c2
        expected = [10, 17]
        assert sorted(squares) == sorted(expected)

    def test_knight_attack_near_edge(self):
        """Test knight attacks near board edge."""
        attacks = generate_knight_attack_bb_from_square(1)  # b1
        squares = get_squares_from_bitboard(attacks)

        # Knight from b1 can attack a3, c3, d2
        expected = [11, 16, 18]
        assert sorted(squares) == sorted(expected)

    def test_knight_attack_all_squares(self):
        """Test that knight attacks are generated for all squares."""
        for square in range(64):
            attacks = generate_knight_attack_bb_from_square(square)
            assert isinstance(attacks, np.uint64)

            # Even corner squares should have some attacks
            squares = get_squares_from_bitboard(attacks)
            if square in [0, 7, 56, 63]:  # corners
                assert len(squares) == 2  # Corners have exactly 2 knight moves
            elif square in [1, 6, 8, 15, 48, 55, 57, 62]:  # near corners
                assert len(squares) == 3  # Near corners have 3 moves
            else:
                assert len(squares) >= 2  # All other squares have at least 2


class TestRookAttacks:
    """Test rook attack pattern generation."""

    def test_rank_attacks_from_center(self):
        """Test rank attacks from center square."""
        attacks = generate_rank_attack_bb_from_square(28)  # e4
        squares = get_squares_from_bitboard(attacks)

        # Rank attacks: north and south along e-file
        expected = [4, 12, 20, 36, 44, 52, 60]  # e1, e2, e3, e5, e6, e7, e8
        assert sorted(squares) == sorted(expected)

    def test_file_attacks_from_center(self):
        """Test file attacks from center square."""
        attacks = generate_file_attack_bb_from_square(28)  # e4
        squares = get_squares_from_bitboard(attacks)

        # File attacks: east and west along 4th rank
        expected = [24, 25, 26, 27, 29, 30, 31]  # a4, b4, c4, d4, f4, g4, h4
        assert sorted(squares) == sorted(expected)

    def test_rook_attacks_combine_rank_and_file(self):
        """Test that rook attacks combine rank and file attacks."""
        square = 28  # e4

        rank_attacks = generate_rank_attack_bb_from_square(square)
        file_attacks = generate_file_attack_bb_from_square(square)
        rook_attacks = generate_rook_attack_bb_from_square(square)

        # Rook attacks should be union of rank and file
        combined = rank_attacks | file_attacks
        assert rook_attacks == combined

    def test_rook_attacks_exclude_starting_square(self):
        """Test that rook attacks don't include starting square."""
        for square in [0, 28, 35, 63]:
            attacks = generate_rook_attack_bb_from_square(square)
            squares = get_squares_from_bitboard(attacks)
            assert square not in squares


class TestBishopAttacks:
    """Test bishop attack pattern generation."""

    def test_bishop_attacks_from_center(self):
        """Test bishop attacks from center square."""
        attacks = generate_diag_attack_bb_from_square(28)  # e4
        squares = get_squares_from_bitboard(attacks)

        # All diagonal squares from e4
        expected = [
            # NE diagonal: f5, g6, h7
            37,
            46,
            55,
            # SW diagonal: d3, c2, b1
            19,
            10,
            1,
            # NW diagonal: d5, c6, b7, a8
            35,
            42,
            49,
            56,
            # SE diagonal: f3, g2, h1
            21,
            14,
            7,
        ]
        assert sorted(squares) == sorted(expected)

    def test_bishop_attacks_from_corner(self):
        """Test bishop attacks from corner square."""
        attacks = generate_diag_attack_bb_from_square(0)  # a1
        squares = get_squares_from_bitboard(attacks)

        # Only one diagonal from a1: b2, c3, d4, e5, f6, g7, h8
        expected = [9, 18, 27, 36, 45, 54, 63]
        assert sorted(squares) == sorted(expected)

    def test_bishop_attacks_exclude_starting_square(self):
        """Test that bishop attacks don't include starting square."""
        for square in [0, 28, 35, 63]:
            attacks = generate_diag_attack_bb_from_square(square)
            squares = get_squares_from_bitboard(attacks)
            assert square not in squares


class TestQueenAttacks:
    """Test queen attack pattern generation."""

    def test_queen_attacks_combine_rook_and_bishop(self):
        """Test that queen attacks combine rook and bishop attacks."""
        square = 28  # e4

        rook_attacks = generate_rook_attack_bb_from_square(square)
        bishop_attacks = generate_diag_attack_bb_from_square(square)
        queen_attacks = generate_queen_attack_bb_from_square(square)

        # Queen attacks should be union of rook and bishop
        combined = rook_attacks | bishop_attacks
        assert queen_attacks == combined

    def test_queen_attacks_from_center(self):
        """Test queen attacks from center square cover maximum squares."""
        attacks = generate_queen_attack_bb_from_square(28)  # e4
        squares = get_squares_from_bitboard(attacks)

        # Queen from e4 should attack 27 squares (8 directions, varying lengths)
        assert len(squares) == 27

        # Should not include the starting square
        assert 28 not in squares


class TestKingAttacks:
    """Test king attack pattern generation."""

    def test_king_attacks_from_center(self):
        """Test king attacks from center square."""
        attacks = generate_king_attack_bb_from_square(28)  # e4
        squares = get_squares_from_bitboard(attacks)

        # King attacks all 8 adjacent squares
        expected = [19, 20, 21, 27, 29, 35, 36, 37]  # All 8 adjacent to e4
        assert sorted(squares) == sorted(expected)
        assert len(squares) == 8

    def test_king_attacks_from_corner(self):
        """Test king attacks from corner square."""
        attacks = generate_king_attack_bb_from_square(0)  # a1
        squares = get_squares_from_bitboard(attacks)

        # King from corner can only attack 3 squares
        expected = [1, 8, 9]  # b1, a2, b2
        assert sorted(squares) == sorted(expected)
        assert len(squares) == 3

    def test_king_attacks_from_edge(self):
        """Test king attacks from edge square."""
        attacks = generate_king_attack_bb_from_square(4)  # e1
        squares = get_squares_from_bitboard(attacks)

        # King from edge can attack 5 squares
        expected = [3, 5, 11, 12, 13]  # d1, f1, d2, e2, f2
        assert sorted(squares) == sorted(expected)
        assert len(squares) == 5


class TestPawnAttacks:
    """Test pawn attack pattern generation."""

    def test_white_pawn_attacks_from_center(self):
        """Test white pawn attacks from center square."""
        attacks = generate_white_pawn_attack_bb_from_square(28)  # e4
        squares = get_squares_from_bitboard(attacks)

        # White pawn attacks diagonally forward (north)
        expected = [35, 37]  # d5, f5
        assert sorted(squares) == sorted(expected)

    def test_white_pawn_attacks_from_edge(self):
        """Test white pawn attacks from file edge."""
        attacks = generate_white_pawn_attack_bb_from_square(24)  # a4
        squares = get_squares_from_bitboard(attacks)

        # White pawn on a-file can only attack one square
        expected = [33]  # b5
        assert sorted(squares) == sorted(expected)

    def test_black_pawn_attacks_from_center(self):
        """Test black pawn attacks from center square."""
        attacks = generate_black_pawn_attack_bb_from_square(28)  # e4
        squares = get_squares_from_bitboard(attacks)

        # Black pawn attacks diagonally forward (south)
        expected = [19, 21]  # d3, f3
        assert sorted(squares) == sorted(expected)

    def test_black_pawn_attacks_from_edge(self):
        """Test black pawn attacks from file edge."""
        attacks = generate_black_pawn_attack_bb_from_square(31)  # h4
        squares = get_squares_from_bitboard(attacks)

        # Black pawn on h-file can only attack one square
        expected = [22]  # g3
        assert sorted(squares) == sorted(expected)


class TestPawnMotion:
    """Test pawn motion pattern generation."""

    def test_white_pawn_motion_from_starting_rank(self):
        """Test white pawn motion from starting rank."""
        motion = generate_white_pawn_motion_bb_from_square(12)  # e2
        squares = get_squares_from_bitboard(motion)

        # White pawn from starting rank can move 1 or 2 squares
        expected = [20, 28]  # e3, e4
        assert sorted(squares) == sorted(expected)

    def test_white_pawn_motion_from_other_rank(self):
        """Test white pawn motion from non-starting rank."""
        motion = generate_white_pawn_motion_bb_from_square(20)  # e3
        squares = get_squares_from_bitboard(motion)

        # White pawn from non-starting rank can only move 1 square
        expected = [28]  # e4
        assert sorted(squares) == sorted(expected)

    def test_black_pawn_motion_from_starting_rank(self):
        """Test black pawn motion from starting rank."""
        motion = generate_black_pawn_motion_bb_from_square(52)  # e7
        squares = get_squares_from_bitboard(motion)

        # Black pawn from starting rank can move 1 or 2 squares
        expected = [44, 36]  # e6, e5
        assert sorted(squares) == sorted(expected)

    def test_black_pawn_motion_from_other_rank(self):
        """Test black pawn motion from non-starting rank."""
        motion = generate_black_pawn_motion_bb_from_square(44)  # e6
        squares = get_squares_from_bitboard(motion)

        # Black pawn from non-starting rank can only move 1 square
        expected = [36]  # e5
        assert sorted(squares) == sorted(expected)


class TestAttackConsistency:
    """Test consistency between different attack patterns."""

    def test_piece_attacks_dont_include_starting_square(self):
        """Test that no attack pattern includes the starting square."""
        test_squares = [0, 7, 28, 35, 56, 63]

        for square in test_squares:
            attacks = [
                generate_knight_attack_bb_from_square(square),
                generate_rook_attack_bb_from_square(square),
                generate_diag_attack_bb_from_square(square),
                generate_queen_attack_bb_from_square(square),
                generate_king_attack_bb_from_square(square),
                generate_white_pawn_attack_bb_from_square(square),
                generate_black_pawn_attack_bb_from_square(square),
            ]

            for attack_bb in attacks:
                squares_attacked = get_squares_from_bitboard(attack_bb)
                assert square not in squares_attacked

    def test_symmetric_pieces_symmetric_attacks(self):
        """Test that symmetric pieces have symmetric attack counts."""
        # Test knight symmetry
        corner_attacks = [
            len(get_squares_from_bitboard(generate_knight_attack_bb_from_square(sq)))
            for sq in [0, 7, 56, 63]  # All corners
        ]
        assert (
            len(set(corner_attacks)) == 1
        )  # All corners should have same number of attacks

        # Test king symmetry
        corner_king_attacks = [
            len(get_squares_from_bitboard(generate_king_attack_bb_from_square(sq)))
            for sq in [0, 7, 56, 63]  # All corners
        ]
        assert len(set(corner_king_attacks)) == 1

    def test_attack_patterns_are_valid_bitboards(self):
        """Test that all attack patterns return valid bitboards."""
        test_squares = [0, 1, 8, 28, 35, 55, 62, 63]

        for square in test_squares:
            attacks = [
                generate_knight_attack_bb_from_square(square),
                generate_rook_attack_bb_from_square(square),
                generate_diag_attack_bb_from_square(square),
                generate_queen_attack_bb_from_square(square),
                generate_king_attack_bb_from_square(square),
                generate_white_pawn_attack_bb_from_square(square),
                generate_black_pawn_attack_bb_from_square(square),
                generate_white_pawn_motion_bb_from_square(square),
                generate_black_pawn_motion_bb_from_square(square),
            ]

            for attack_bb in attacks:
                assert isinstance(attack_bb, np.uint64)
                # All attacks should be within valid board range
                squares_attacked = get_squares_from_bitboard(attack_bb)
                for sq in squares_attacked:
                    assert 0 <= sq < 64


class TestAttackBoundaries:
    """Test attack patterns at board boundaries."""

    def test_attacks_respect_board_boundaries(self):
        """Test that attacks don't go outside the board."""
        # Test all edge squares
        edge_squares = (
            list(range(0, 8))  # Rank 1
            + list(range(56, 64))  # Rank 8
            + [8, 16, 24, 32, 40, 48]  # A file
            + [15, 23, 31, 39, 47, 55]  # H file
        )

        for square in edge_squares:
            attacks = [
                generate_knight_attack_bb_from_square(square),
                generate_king_attack_bb_from_square(square),
                generate_white_pawn_attack_bb_from_square(square),
                generate_black_pawn_attack_bb_from_square(square),
            ]

            for attack_bb in attacks:
                squares_attacked = get_squares_from_bitboard(attack_bb)
                for sq in squares_attacked:
                    assert 0 <= sq < 64

    def test_wrapping_prevention(self):
        """Test that attacks don't wrap around board edges."""
        # Test specific cases that might wrap

        # Knight from h1 shouldn't attack outside board
        knight_attacks = generate_knight_attack_bb_from_square(7)  # h1
        squares = get_squares_from_bitboard(knight_attacks)
        # Should only be able to attack f2 and g3
        assert len(squares) == 2
        assert all(sq in [13, 22] for sq in squares)  # f2, g3

        # King from h8 shouldn't wrap
        king_attacks = generate_king_attack_bb_from_square(63)  # h8
        squares = get_squares_from_bitboard(king_attacks)
        # Should only attack g8, g7, h7
        assert len(squares) == 3
        assert all(sq in [62, 54, 55] for sq in squares)  # g8, g7, h7


class TestSpecialCases:
    """Test special cases and edge conditions."""

    def test_pawn_motion_boundary_cases(self):
        """Test pawn motion at rank boundaries."""
        # White pawn on 8th rank (shouldn't happen in real game, but test anyway)
        motion = generate_white_pawn_motion_bb_from_square(60)  # e8
        squares = get_squares_from_bitboard(motion)
        assert len(squares) == 0  # No legal moves beyond 8th rank

        # Black pawn on 1st rank
        motion = generate_black_pawn_motion_bb_from_square(4)  # e1
        squares = get_squares_from_bitboard(motion)
        assert len(squares) == 0  # No legal moves beyond 1st rank

    def test_maximum_attack_counts(self):
        """Test maximum possible attack counts for each piece type."""
        # Queen from center should have maximum attacks
        queen_attacks = generate_queen_attack_bb_from_square(28)  # e4
        queen_count = len(get_squares_from_bitboard(queen_attacks))

        # Knight can attack at most 8 squares
        max_knight = 0
        for sq in range(64):
            knight_attacks = generate_knight_attack_bb_from_square(sq)
            count = len(get_squares_from_bitboard(knight_attacks))
            max_knight = max(max_knight, count)
        assert max_knight == 8

        # King can attack at most 8 squares
        max_king = 0
        for sq in range(64):
            king_attacks = generate_king_attack_bb_from_square(sq)
            count = len(get_squares_from_bitboard(king_attacks))
            max_king = max(max_king, count)
        assert max_king == 8
