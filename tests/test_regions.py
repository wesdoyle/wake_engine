"""
Test suite for bitboard/regions.py - Board region access functions.

Tests the definitions of chess board regions including ranks, files,
and special regions like center squares, flanks, and castling sides.
"""

import numpy as np

from wake.bitboard.core import get_squares_from_bitboard
from wake.bitboard.regions import (
    rank_8_bb,
    rank_7_bb,
    rank_6_bb,
    rank_5_bb,
    rank_4_bb,
    rank_3_bb,
    rank_2_bb,
    rank_1_bb,
    file_h_bb,
    file_g_bb,
    file_f_bb,
    file_e_bb,
    file_d_bb,
    file_c_bb,
    file_b_bb,
    file_a_bb,
    dark_squares_bb,
    light_squares_bb,
    center_squares_bb,
    flanks_bb,
    center_files_bb,
    kingside_bb,
    queenside_bb,
)
from wake.constants import Square


class TestRankRegions:
    """Test rank bitboard functions."""

    def test_rank_1_definition(self):
        """Test rank 1 bitboard contains correct squares."""
        rank_1_squares = get_squares_from_bitboard(rank_1_bb())
        expected_squares = {0, 1, 2, 3, 4, 5, 6, 7}  # a1-h1
        assert set(rank_1_squares) == expected_squares
        assert len(rank_1_squares) == 8

    def test_rank_2_definition(self):
        """Test rank 2 bitboard contains correct squares."""
        rank_2_squares = get_squares_from_bitboard(rank_2_bb())
        expected_squares = {8, 9, 10, 11, 12, 13, 14, 15}  # a2-h2
        assert set(rank_2_squares) == expected_squares
        assert len(rank_2_squares) == 8

    def test_rank_3_definition(self):
        """Test rank 3 bitboard contains correct squares."""
        rank_3_squares = get_squares_from_bitboard(rank_3_bb())
        expected_squares = {16, 17, 18, 19, 20, 21, 22, 23}  # a3-h3
        assert set(rank_3_squares) == expected_squares
        assert len(rank_3_squares) == 8

    def test_rank_4_definition(self):
        """Test rank 4 bitboard contains correct squares."""
        rank_4_squares = get_squares_from_bitboard(rank_4_bb())
        expected_squares = {24, 25, 26, 27, 28, 29, 30, 31}  # a4-h4
        assert set(rank_4_squares) == expected_squares
        assert len(rank_4_squares) == 8

    def test_rank_5_definition(self):
        """Test rank 5 bitboard contains correct squares."""
        rank_5_squares = get_squares_from_bitboard(rank_5_bb())
        expected_squares = {32, 33, 34, 35, 36, 37, 38, 39}  # a5-h5
        assert set(rank_5_squares) == expected_squares
        assert len(rank_5_squares) == 8

    def test_rank_6_definition(self):
        """Test rank 6 bitboard contains correct squares."""
        rank_6_squares = get_squares_from_bitboard(rank_6_bb())
        expected_squares = {40, 41, 42, 43, 44, 45, 46, 47}  # a6-h6
        assert set(rank_6_squares) == expected_squares
        assert len(rank_6_squares) == 8

    def test_rank_7_definition(self):
        """Test rank 7 bitboard contains correct squares."""
        rank_7_squares = get_squares_from_bitboard(rank_7_bb())
        expected_squares = {48, 49, 50, 51, 52, 53, 54, 55}  # a7-h7
        assert set(rank_7_squares) == expected_squares
        assert len(rank_7_squares) == 8

    def test_rank_8_definition(self):
        """Test rank 8 bitboard contains correct squares."""
        rank_8_squares = get_squares_from_bitboard(rank_8_bb())
        expected_squares = {56, 57, 58, 59, 60, 61, 62, 63}  # a8-h8
        assert set(rank_8_squares) == expected_squares
        assert len(rank_8_squares) == 8

    def test_ranks_are_disjoint(self):
        """Test that all ranks are mutually exclusive."""
        ranks = [
            rank_1_bb(),
            rank_2_bb(),
            rank_3_bb(),
            rank_4_bb(),
            rank_5_bb(),
            rank_6_bb(),
            rank_7_bb(),
            rank_8_bb(),
        ]

        for i, rank_i in enumerate(ranks):
            for j, rank_j in enumerate(ranks):
                if i != j:
                    # No overlap between different ranks
                    assert (rank_i & rank_j) == 0

    def test_ranks_cover_entire_board(self):
        """Test that all ranks together cover the entire board."""
        all_ranks = (
            rank_1_bb()
            | rank_2_bb()
            | rank_3_bb()
            | rank_4_bb()
            | rank_5_bb()
            | rank_6_bb()
            | rank_7_bb()
            | rank_8_bb()
        )

        all_squares = get_squares_from_bitboard(all_ranks)
        assert len(all_squares) == 64
        assert set(all_squares) == set(range(64))


class TestFileRegions:
    """Test file bitboard functions."""

    def test_file_a_definition(self):
        """Test file A bitboard contains correct squares."""
        file_a_squares = get_squares_from_bitboard(file_a_bb())
        expected_squares = {0, 8, 16, 24, 32, 40, 48, 56}  # a1-a8
        assert set(file_a_squares) == expected_squares
        assert len(file_a_squares) == 8

    def test_file_b_definition(self):
        """Test file B bitboard contains correct squares."""
        file_b_squares = get_squares_from_bitboard(file_b_bb())
        expected_squares = {1, 9, 17, 25, 33, 41, 49, 57}  # b1-b8
        assert set(file_b_squares) == expected_squares
        assert len(file_b_squares) == 8

    def test_file_c_definition(self):
        """Test file C bitboard contains correct squares."""
        file_c_squares = get_squares_from_bitboard(file_c_bb())
        expected_squares = {2, 10, 18, 26, 34, 42, 50, 58}  # c1-c8
        assert set(file_c_squares) == expected_squares
        assert len(file_c_squares) == 8

    def test_file_d_definition(self):
        """Test file D bitboard contains correct squares."""
        file_d_squares = get_squares_from_bitboard(file_d_bb())
        expected_squares = {3, 11, 19, 27, 35, 43, 51, 59}  # d1-d8
        assert set(file_d_squares) == expected_squares
        assert len(file_d_squares) == 8

    def test_file_e_definition(self):
        """Test file E bitboard contains correct squares."""
        file_e_squares = get_squares_from_bitboard(file_e_bb())
        expected_squares = {4, 12, 20, 28, 36, 44, 52, 60}  # e1-e8
        assert set(file_e_squares) == expected_squares
        assert len(file_e_squares) == 8

    def test_file_f_definition(self):
        """Test file F bitboard contains correct squares."""
        file_f_squares = get_squares_from_bitboard(file_f_bb())
        expected_squares = {5, 13, 21, 29, 37, 45, 53, 61}  # f1-f8
        assert set(file_f_squares) == expected_squares
        assert len(file_f_squares) == 8

    def test_file_g_definition(self):
        """Test file G bitboard contains correct squares."""
        file_g_squares = get_squares_from_bitboard(file_g_bb())
        expected_squares = {6, 14, 22, 30, 38, 46, 54, 62}  # g1-g8
        assert set(file_g_squares) == expected_squares
        assert len(file_g_squares) == 8

    def test_file_h_definition(self):
        """Test file H bitboard contains correct squares."""
        file_h_squares = get_squares_from_bitboard(file_h_bb())
        expected_squares = {7, 15, 23, 31, 39, 47, 55, 63}  # h1-h8
        assert set(file_h_squares) == expected_squares
        assert len(file_h_squares) == 8

    def test_files_are_disjoint(self):
        """Test that all files are mutually exclusive."""
        files = [
            file_a_bb(),
            file_b_bb(),
            file_c_bb(),
            file_d_bb(),
            file_e_bb(),
            file_f_bb(),
            file_g_bb(),
            file_h_bb(),
        ]

        for i, file_i in enumerate(files):
            for j, file_j in enumerate(files):
                if i != j:
                    # No overlap between different files
                    assert (file_i & file_j) == 0

    def test_files_cover_entire_board(self):
        """Test that all files together cover the entire board."""
        all_files = (
            file_a_bb()
            | file_b_bb()
            | file_c_bb()
            | file_d_bb()
            | file_e_bb()
            | file_f_bb()
            | file_g_bb()
            | file_h_bb()
        )

        all_squares = get_squares_from_bitboard(all_files)
        assert len(all_squares) == 64
        assert set(all_squares) == set(range(64))


class TestSquareColorRegions:
    """Test square color bitboard functions."""

    def test_dark_squares_definition(self):
        """Test dark squares bitboard contains correct squares."""
        dark_squares = get_squares_from_bitboard(dark_squares_bb())

        # Dark squares: a1, c1, e1, g1, b2, d2, f2, h2, etc.
        # Pattern: squares where (rank + file) is even (corrected)
        expected_dark = []
        for square in range(64):
            rank = square // 8
            file = square % 8
            if (rank + file) % 2 == 0:
                expected_dark.append(square)

        assert set(dark_squares) == set(expected_dark)
        assert len(dark_squares) == 32

    def test_light_squares_definition(self):
        """Test light squares bitboard contains correct squares."""
        light_squares = get_squares_from_bitboard(light_squares_bb())

        # Light squares: b1, d1, f1, h1, a2, c2, e2, g2, etc.
        # Pattern: squares where (rank + file) is odd (corrected)
        expected_light = []
        for square in range(64):
            rank = square // 8
            file = square % 8
            if (rank + file) % 2 == 1:
                expected_light.append(square)

        assert set(light_squares) == set(expected_light)
        assert len(light_squares) == 32

    def test_dark_and_light_squares_are_disjoint(self):
        """Test that dark and light squares don't overlap."""
        dark_bb = dark_squares_bb()
        light_bb = light_squares_bb()

        assert (dark_bb & light_bb) == 0

    def test_dark_and_light_squares_cover_board(self):
        """Test that dark and light squares together cover the entire board."""
        all_colored_squares = dark_squares_bb() | light_squares_bb()

        all_squares = get_squares_from_bitboard(all_colored_squares)
        assert len(all_squares) == 64
        assert set(all_squares) == set(range(64))


class TestSpecialRegions:
    """Test special region bitboard functions."""

    def test_center_squares_definition(self):
        """Test center squares bitboard contains correct squares."""
        center_squares = get_squares_from_bitboard(center_squares_bb())

        # Center squares: d4, d5, e4, e5 (squares 27, 28, 35, 36)
        expected_center = {27, 28, 35, 36}  # d4, e4, d5, e5
        assert set(center_squares) == expected_center
        assert len(center_squares) == 4

    def test_flanks_definition(self):
        """Test flanks bitboard contains correct squares."""
        flanks_squares = get_squares_from_bitboard(flanks_bb())

        # Flanks: A file + H file
        expected_flanks = {0, 7, 8, 15, 16, 23, 24, 31, 32, 39, 40, 47, 48, 55, 56, 63}
        assert set(flanks_squares) == expected_flanks
        assert len(flanks_squares) == 16

    def test_center_files_definition(self):
        """Test center files bitboard contains correct squares."""
        center_files_squares = get_squares_from_bitboard(center_files_bb())

        # Center files: C, D, E, F files
        expected_center_files = set()
        for file_squares in [
            get_squares_from_bitboard(file_c_bb()),
            get_squares_from_bitboard(file_d_bb()),
            get_squares_from_bitboard(file_e_bb()),
            get_squares_from_bitboard(file_f_bb()),
        ]:
            expected_center_files.update(file_squares)

        assert set(center_files_squares) == expected_center_files
        assert len(center_files_squares) == 32

    def test_kingside_definition(self):
        """Test kingside bitboard contains correct squares."""
        kingside_squares = get_squares_from_bitboard(kingside_bb())

        # Kingside: E, F, G, H files
        expected_kingside = set()
        for file_squares in [
            get_squares_from_bitboard(file_e_bb()),
            get_squares_from_bitboard(file_f_bb()),
            get_squares_from_bitboard(file_g_bb()),
            get_squares_from_bitboard(file_h_bb()),
        ]:
            expected_kingside.update(file_squares)

        assert set(kingside_squares) == expected_kingside
        assert len(kingside_squares) == 32

    def test_queenside_definition(self):
        """Test queenside bitboard contains correct squares."""
        queenside_squares = get_squares_from_bitboard(queenside_bb())

        # Queenside: A, B, C, D files
        expected_queenside = set()
        for file_squares in [
            get_squares_from_bitboard(file_a_bb()),
            get_squares_from_bitboard(file_b_bb()),
            get_squares_from_bitboard(file_c_bb()),
            get_squares_from_bitboard(file_d_bb()),
        ]:
            expected_queenside.update(file_squares)

        assert set(queenside_squares) == expected_queenside
        assert len(queenside_squares) == 32

    def test_kingside_and_queenside_are_disjoint(self):
        """Test that kingside and queenside don't overlap."""
        kingside_bb_val = kingside_bb()
        queenside_bb_val = queenside_bb()

        assert (kingside_bb_val & queenside_bb_val) == 0

    def test_kingside_and_queenside_cover_board(self):
        """Test that kingside and queenside together cover the entire board."""
        all_sides = kingside_bb() | queenside_bb()

        all_squares = get_squares_from_bitboard(all_sides)
        assert len(all_squares) == 64
        assert set(all_squares) == set(range(64))


class TestRegionConsistency:
    """Test consistency between different region definitions."""

    def test_region_functions_return_uint64(self):
        """Test that all region functions return np.uint64."""
        region_functions = [
            rank_1_bb,
            rank_2_bb,
            rank_3_bb,
            rank_4_bb,
            rank_5_bb,
            rank_6_bb,
            rank_7_bb,
            rank_8_bb,
            file_a_bb,
            file_b_bb,
            file_c_bb,
            file_d_bb,
            file_e_bb,
            file_f_bb,
            file_g_bb,
            file_h_bb,
            dark_squares_bb,
            light_squares_bb,
            center_squares_bb,
            flanks_bb,
            center_files_bb,
            kingside_bb,
            queenside_bb,
        ]

        for region_func in region_functions:
            result = region_func()
            assert isinstance(result, np.uint64)

    def test_center_squares_are_subset_of_center_files(self):
        """Test that center squares are contained within center files."""
        center_squares = center_squares_bb()
        center_files = center_files_bb()

        # Center squares should be entirely contained within center files
        assert (center_squares & center_files) == center_squares

    def test_flanks_and_center_files_are_complementary(self):
        """Test that flanks and center files are complementary."""
        flanks = flanks_bb()
        center_files = center_files_bb()

        # Should be disjoint
        assert (flanks & center_files) == 0

        # Note: flanks (A, H) + center files (C, D, E, F) = 6 files out of 8
        # They don't cover entire board (missing B and G files)
        all_regions = flanks | center_files
        all_squares = get_squares_from_bitboard(all_regions)
        assert len(all_squares) == 48  # 6 files * 8 squares per file

    def test_specific_square_memberships(self):
        """Test specific squares belong to expected regions."""
        # Test E4 (square 28)
        e4_square = np.uint64(1) << np.uint64(28)

        # E4 should be in center squares
        assert (e4_square & center_squares_bb()) != 0

        # E4 should be in center files
        assert (e4_square & center_files_bb()) != 0

        # E4 should be in kingside
        assert (e4_square & kingside_bb()) != 0

        # E4 should be in rank 4 and file E
        assert (e4_square & rank_4_bb()) != 0
        assert (e4_square & file_e_bb()) != 0

    def test_region_intersections_make_sense(self):
        """Test that region intersections produce logical results."""
        # Intersection of rank 1 and file A should be A1
        a1_intersection = rank_1_bb() & file_a_bb()
        a1_squares = get_squares_from_bitboard(a1_intersection)
        assert set(a1_squares) == {0}  # A1 is square 0

        # Intersection of rank 8 and file H should be H8
        h8_intersection = rank_8_bb() & file_h_bb()
        h8_squares = get_squares_from_bitboard(h8_intersection)
        assert set(h8_squares) == {63}  # H8 is square 63

        # Intersection of center squares and rank 4 should be D4 and E4
        center_rank4_intersection = center_squares_bb() & rank_4_bb()
        center_rank4_squares = get_squares_from_bitboard(center_rank4_intersection)
        assert set(center_rank4_squares) == {27, 28}  # D4, E4


class TestRegionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_regions_if_any(self):
        """Test handling of potentially empty regions."""
        # All current regions should be non-empty
        region_functions = [
            rank_1_bb,
            rank_2_bb,
            rank_3_bb,
            rank_4_bb,
            rank_5_bb,
            rank_6_bb,
            rank_7_bb,
            rank_8_bb,
            file_a_bb,
            file_b_bb,
            file_c_bb,
            file_d_bb,
            file_e_bb,
            file_f_bb,
            file_g_bb,
            file_h_bb,
            dark_squares_bb,
            light_squares_bb,
            center_squares_bb,
            flanks_bb,
            center_files_bb,
            kingside_bb,
            queenside_bb,
        ]

        for region_func in region_functions:
            result = region_func()
            assert result != 0  # No region should be empty

    def test_region_bounds_checking(self):
        """Test that all region squares are within board bounds."""
        region_functions = [
            rank_1_bb,
            rank_2_bb,
            rank_3_bb,
            rank_4_bb,
            rank_5_bb,
            rank_6_bb,
            rank_7_bb,
            rank_8_bb,
            file_a_bb,
            file_b_bb,
            file_c_bb,
            file_d_bb,
            file_e_bb,
            file_f_bb,
            file_g_bb,
            file_h_bb,
            dark_squares_bb,
            light_squares_bb,
            center_squares_bb,
            flanks_bb,
            center_files_bb,
            kingside_bb,
            queenside_bb,
        ]

        for region_func in region_functions:
            result = region_func()
            squares = get_squares_from_bitboard(result)

            # All squares should be in valid range
            for square in squares:
                assert 0 <= square <= 63

    def test_region_consistency_with_constants(self):
        """Test that regions are consistent with chess constants."""
        # Test that rank 1 contains the expected starting squares
        rank_1_squares = set(get_squares_from_bitboard(rank_1_bb()))

        # Should contain all pieces' starting squares
        expected_rank_1_pieces = {
            Square.A1,
            Square.B1,
            Square.C1,
            Square.D1,
            Square.E1,
            Square.F1,
            Square.G1,
            Square.H1,
        }

        # Convert Square constants to integers for comparison
        expected_rank_1_ints = {int(square) for square in expected_rank_1_pieces}
        assert rank_1_squares == expected_rank_1_ints
