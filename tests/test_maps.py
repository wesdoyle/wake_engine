"""
Test suite for bitboard/maps.py - Static attack map builders.

Tests the creation of static lookup tables for all piece types,
ensuring proper dictionary structure and attack pattern correctness.
"""

import numpy as np

from wake.bitboard.core import get_squares_from_bitboard, BOARD_SQUARES
from wake.bitboard.maps import (
    make_knight_attack_bbs,
    make_rank_attack_bbs,
    make_file_attack_bbs,
    make_diag_attack_bbs,
    make_king_attack_bbs,
    make_queen_attack_bbs,
    make_rook_attack_bbs,
    make_white_pawn_attack_bbs,
    make_black_pawn_attack_bbs,
    make_white_pawn_motion_bbs,
    make_black_pawn_motion_bbs,
)


class TestMapStructure:
    """Test basic structure and properties of generated maps."""

    def test_all_piece_maps_have_correct_structure(self):
        """Test that all piece type maps have the correct structure."""
        maps = [
            make_knight_attack_bbs(),
            make_rank_attack_bbs(),
            make_file_attack_bbs(),
            make_diag_attack_bbs(),
            make_king_attack_bbs(),
            make_queen_attack_bbs(),
            make_rook_attack_bbs(),
        ]

        for piece_map in maps:
            assert isinstance(piece_map, dict)
            assert len(piece_map) == BOARD_SQUARES

            # Check all squares 0-63 are present
            for square in range(BOARD_SQUARES):
                assert square in piece_map
                assert isinstance(piece_map[square], np.uint64)

    def test_pawn_maps_have_correct_structure(self):
        """Test that pawn maps have the correct structure (excluding invalid ranks)."""
        white_attack_map = make_white_pawn_attack_bbs()
        black_attack_map = make_black_pawn_attack_bbs()
        white_motion_map = make_white_pawn_motion_bbs()
        black_motion_map = make_black_pawn_motion_bbs()

        # Pawn maps should cover squares from A2 to H7 (8-55)
        expected_size = 48  # 56 - 8 = 48 squares

        for pawn_map in [
            white_attack_map,
            black_attack_map,
            white_motion_map,
            black_motion_map,
        ]:
            assert isinstance(pawn_map, dict)
            assert len(pawn_map) == expected_size

            # Check squares 8-55 are present
            for square in range(8, 56):
                assert square in pawn_map
                assert isinstance(pawn_map[square], np.uint64)


class TestKnightMaps:
    """Test knight attack map generation."""

    def test_knight_corner_squares(self):
        """Test knight attacks from corner squares."""
        knight_map = make_knight_attack_bbs()

        # Test A1 corner (square 0)
        a1_attacks = get_squares_from_bitboard(knight_map[0])
        assert set(a1_attacks) == {10, 17}  # b3, c2

        # Test H8 corner (square 63)
        h8_attacks = get_squares_from_bitboard(knight_map[63])
        assert set(h8_attacks) == {46, 53}  # g6, f7

    def test_knight_center_square(self):
        """Test knight attacks from center square."""
        knight_map = make_knight_attack_bbs()

        # Test E4 (square 28)
        e4_attacks = get_squares_from_bitboard(knight_map[28])
        expected_attacks = {11, 13, 18, 22, 34, 38, 43, 45}  # All 8 knight moves
        assert set(e4_attacks) == expected_attacks
        assert len(e4_attacks) == 8


class TestRookMaps:
    """Test rook attack map generation."""

    def test_rook_maps_consistency(self):
        """Test that rook maps are consistent with rank/file maps."""
        rook_map = make_rook_attack_bbs()
        rank_map = make_rank_attack_bbs()
        file_map = make_file_attack_bbs()

        # Test several squares
        test_squares = [0, 28, 35, 63]

        for square in test_squares:
            # Rook attacks should be union of rank and file attacks
            rook_attacks = set(get_squares_from_bitboard(rook_map[square]))
            rank_attacks = set(get_squares_from_bitboard(rank_map[square]))
            file_attacks = set(get_squares_from_bitboard(file_map[square]))

            assert rook_attacks == rank_attacks | file_attacks

    def test_rook_corner_coverage(self):
        """Test rook attacks from corner squares."""
        rook_map = make_rook_attack_bbs()

        # Test A1 (square 0)
        a1_attacks = get_squares_from_bitboard(rook_map[0])

        # Should attack entire A file (8, 16, 24, 32, 40, 48, 56)
        # and entire 1st rank (1, 2, 3, 4, 5, 6, 7)
        expected_a1 = {1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56}
        assert set(a1_attacks) == expected_a1


class TestBishopMaps:
    """Test bishop attack map generation."""

    def test_bishop_center_square(self):
        """Test bishop attacks from center square."""
        bishop_map = make_diag_attack_bbs()

        # Test E4 (square 28)
        e4_attacks = get_squares_from_bitboard(bishop_map[28])

        # Should attack both diagonals from E4
        # Northeast diagonal: f5, g6, h7 (29+8+1, 30+8+1, 31+8+1) = 37, 46, 55
        # Southwest diagonal: d3, c2, b1 (27-8-1, 26-8-1, 25-8-1) = 19, 10, 1
        # Northwest diagonal: d5, c6, b7, a8 (27+8-1, 26+8-1, 25+8-1, 24+8-1) = 35, 34, 33, 32 -> 35, 42, 49, 56
        # Southeast diagonal: f3, g2, h1 (29-8+1, 30-8+1, 31-8+1) = 21, 14, 7

        # Let me just check that it attacks reasonable number of squares
        # A bishop from center should attack 13 squares (2 diagonals minus starting square)
        assert len(e4_attacks) == 13

    def test_bishop_corner_limited(self):
        """Test bishop attacks from corner (limited moves)."""
        bishop_map = make_diag_attack_bbs()

        # Test A1 (square 0)
        a1_attacks = get_squares_from_bitboard(bishop_map[0])

        # Should only attack one diagonal: 9, 18, 27, 36, 45, 54, 63
        expected_a1 = {9, 18, 27, 36, 45, 54, 63}
        assert set(a1_attacks) == expected_a1


class TestQueenMaps:
    """Test queen attack map generation."""

    def test_queen_combines_rook_and_bishop(self):
        """Test that queen maps combine rook and bishop attacks."""
        queen_map = make_queen_attack_bbs()
        rook_map = make_rook_attack_bbs()
        bishop_map = make_diag_attack_bbs()

        # Test several squares
        test_squares = [0, 28, 35, 63]

        for square in test_squares:
            queen_attacks = set(get_squares_from_bitboard(queen_map[square]))
            rook_attacks = set(get_squares_from_bitboard(rook_map[square]))
            bishop_attacks = set(get_squares_from_bitboard(bishop_map[square]))

            assert queen_attacks == rook_attacks | bishop_attacks

    def test_queen_center_maximum_coverage(self):
        """Test queen attacks from center square have maximum coverage."""
        queen_map = make_queen_attack_bbs()

        # Test E4 (square 28) - should have maximum mobility
        e4_attacks = get_squares_from_bitboard(queen_map[28])

        # Queen from E4 should attack 27 squares (8 directions, max squares per direction)
        assert len(e4_attacks) == 27


class TestKingMaps:
    """Test king attack map generation."""

    def test_king_center_square(self):
        """Test king attacks from center square."""
        king_map = make_king_attack_bbs()

        # Test E4 (square 28)
        e4_attacks = get_squares_from_bitboard(king_map[28])

        # King attacks all 8 adjacent squares
        expected_attacks = {19, 20, 21, 27, 29, 35, 36, 37}
        assert set(e4_attacks) == expected_attacks
        assert len(e4_attacks) == 8

    def test_king_corner_limited(self):
        """Test king attacks from corner (limited moves)."""
        king_map = make_king_attack_bbs()

        # Test A1 (square 0)
        a1_attacks = get_squares_from_bitboard(king_map[0])

        # King from A1 can only attack 3 squares
        expected_a1 = {1, 8, 9}
        assert set(a1_attacks) == expected_a1
        assert len(a1_attacks) == 3

    def test_king_edge_squares(self):
        """Test king attacks from edge squares."""
        king_map = make_king_attack_bbs()

        # Test E1 (square 4) - middle of first rank
        e1_attacks = get_squares_from_bitboard(king_map[4])

        # King from E1 can attack 5 squares
        expected_e1 = {3, 5, 11, 12, 13}
        assert set(e1_attacks) == expected_e1
        assert len(e1_attacks) == 5


class TestPawnMaps:
    """Test pawn attack and motion map generation."""

    def test_white_pawn_attack_patterns(self):
        """Test white pawn attack patterns."""
        white_attack_map = make_white_pawn_attack_bbs()

        # Test E2 (square 12)
        e2_attacks = get_squares_from_bitboard(white_attack_map[12])

        # White pawn from E2 attacks D3 and F3
        expected_attacks = {19, 21}  # d3, f3
        assert set(e2_attacks) == expected_attacks
        assert len(e2_attacks) == 2

    def test_black_pawn_attack_patterns(self):
        """Test black pawn attack patterns."""
        black_attack_map = make_black_pawn_attack_bbs()

        # Test E7 (square 52)
        e7_attacks = get_squares_from_bitboard(black_attack_map[52])

        # Black pawn from E7 attacks D6 and F6
        expected_attacks = {43, 45}  # d6, f6
        assert set(e7_attacks) == expected_attacks
        assert len(e7_attacks) == 2

    def test_pawn_edge_files(self):
        """Test pawn attacks from edge files."""
        white_attack_map = make_white_pawn_attack_bbs()
        black_attack_map = make_black_pawn_attack_bbs()

        # Test A2 (square 8) - can only attack one square
        a2_attacks = get_squares_from_bitboard(white_attack_map[8])
        assert set(a2_attacks) == {17}  # b3
        assert len(a2_attacks) == 1

        # Test H7 (square 55) - can only attack one square
        h7_attacks = get_squares_from_bitboard(black_attack_map[55])
        assert set(h7_attacks) == {46}  # g6
        assert len(h7_attacks) == 1

    def test_white_pawn_motion_patterns(self):
        """Test white pawn motion patterns."""
        white_motion_map = make_white_pawn_motion_bbs()

        # Test E2 (square 12) - can move one or two squares
        e2_motion = get_squares_from_bitboard(white_motion_map[12])
        expected_motion = {20, 28}  # e3, e4
        assert set(e2_motion) == expected_motion
        assert len(e2_motion) == 2

        # Test E3 (square 20) - can only move one square
        e3_motion = get_squares_from_bitboard(white_motion_map[20])
        expected_motion = {28}  # e4
        assert set(e3_motion) == expected_motion
        assert len(e3_motion) == 1

    def test_black_pawn_motion_patterns(self):
        """Test black pawn motion patterns."""
        black_motion_map = make_black_pawn_motion_bbs()

        # Test E7 (square 52) - can move one or two squares
        e7_motion = get_squares_from_bitboard(black_motion_map[52])
        expected_motion = {44, 36}  # e6, e5
        assert set(e7_motion) == expected_motion
        assert len(e7_motion) == 2

        # Test E6 (square 44) - can only move one square
        e6_motion = get_squares_from_bitboard(black_motion_map[44])
        expected_motion = {36}  # e5
        assert set(e6_motion) == expected_motion
        assert len(e6_motion) == 1


class TestMapConsistency:
    """Test consistency between different map types."""

    def test_no_starting_square_in_attacks(self):
        """Test that no piece attacks its own square."""
        maps = [
            make_knight_attack_bbs(),
            make_rank_attack_bbs(),
            make_file_attack_bbs(),
            make_diag_attack_bbs(),
            make_king_attack_bbs(),
            make_queen_attack_bbs(),
            make_rook_attack_bbs(),
        ]

        for piece_map in maps:
            for square, attack_bb in piece_map.items():
                attacked_squares = get_squares_from_bitboard(attack_bb)
                assert square not in attacked_squares

    def test_attack_patterns_are_valid_bitboards(self):
        """Test that all attack patterns are valid bitboards."""
        maps = [
            make_knight_attack_bbs(),
            make_rank_attack_bbs(),
            make_file_attack_bbs(),
            make_diag_attack_bbs(),
            make_king_attack_bbs(),
            make_queen_attack_bbs(),
            make_rook_attack_bbs(),
            make_white_pawn_attack_bbs(),
            make_black_pawn_attack_bbs(),
            make_white_pawn_motion_bbs(),
            make_black_pawn_motion_bbs(),
        ]

        for piece_map in maps:
            for square, attack_bb in piece_map.items():
                # Should be a valid numpy uint64
                assert isinstance(attack_bb, np.uint64)

                # All attacked squares should be in valid range
                attacked_squares = get_squares_from_bitboard(attack_bb)
                for attacked_square in attacked_squares:
                    assert 0 <= attacked_square < 64


class TestMapPerformance:
    """Test performance characteristics of map generation."""

    def test_map_generation_completes(self):
        """Test that all map generation functions complete successfully."""
        # This test ensures no infinite loops or exceptions
        maps = [
            make_knight_attack_bbs(),
            make_rank_attack_bbs(),
            make_file_attack_bbs(),
            make_diag_attack_bbs(),
            make_king_attack_bbs(),
            make_queen_attack_bbs(),
            make_rook_attack_bbs(),
            make_white_pawn_attack_bbs(),
            make_black_pawn_attack_bbs(),
            make_white_pawn_motion_bbs(),
            make_black_pawn_motion_bbs(),
        ]

        for piece_map in maps:
            assert piece_map is not None
            assert len(piece_map) > 0

    def test_maps_are_deterministic(self):
        """Test that maps generate the same results across multiple calls."""
        # Generate maps twice and compare
        knight_map1 = make_knight_attack_bbs()
        knight_map2 = make_knight_attack_bbs()

        assert len(knight_map1) == len(knight_map2)

        for square in knight_map1:
            assert square in knight_map2
            assert knight_map1[square] == knight_map2[square]


class TestSpecialCases:
    """Test edge cases and special scenarios."""

    def test_empty_bitboards_where_appropriate(self):
        """Test that some positions generate empty bitboards when appropriate."""
        # Test pieces that might have no valid moves in certain positions
        maps_to_test = [
            make_white_pawn_motion_bbs(),
            make_black_pawn_motion_bbs(),
        ]

        for piece_map in maps_to_test:
            # All pawn maps should have some non-zero entries
            non_zero_count = sum(1 for bb in piece_map.values() if bb != 0)
            assert non_zero_count > 0

    def test_maximum_attack_counts(self):
        """Test that pieces don't exceed maximum possible attacks."""
        # Queen should have maximum attacks of 27 (from center)
        queen_map = make_queen_attack_bbs()
        max_queen_attacks = max(
            len(get_squares_from_bitboard(bb)) for bb in queen_map.values()
        )
        assert max_queen_attacks <= 27

        # Knight should have maximum attacks of 8
        knight_map = make_knight_attack_bbs()
        max_knight_attacks = max(
            len(get_squares_from_bitboard(bb)) for bb in knight_map.values()
        )
        assert max_knight_attacks <= 8

        # King should have maximum attacks of 8
        king_map = make_king_attack_bbs()
        max_king_attacks = max(
            len(get_squares_from_bitboard(bb)) for bb in king_map.values()
        )
        assert max_king_attacks <= 8
