"""
Tests for the bitboard rays module.

Tests all ray generation functions for sliding pieces including
north, south, east, west, and diagonal rays.
"""

import numpy as np

from wake.bitboard.core import make_uint64, get_squares_from_bitboard
from wake.bitboard.rays import (
    get_south_ray,
    get_north_ray,
    get_west_ray,
    get_east_ray,
    get_southeast_ray,
    get_northwest_ray,
    get_southwest_ray,
    get_northeast_ray,
)


class TestStraightRays:
    """Test straight-line ray generation (N, S, E, W)."""

    def test_north_ray_from_center(self):
        """Test north ray from center square."""
        bb = make_uint64()
        result = get_north_ray(bb, 28)  # e4

        squares = get_squares_from_bitboard(result)
        expected = [36, 44, 52, 60]  # e5, e6, e7, e8
        assert sorted(squares) == sorted(expected)

    def test_north_ray_from_edge(self):
        """Test north ray from top edge (should be empty)."""
        bb = make_uint64()
        result = get_north_ray(bb, 60)  # e8

        squares = get_squares_from_bitboard(result)
        assert squares == []  # No squares north of top rank

    def test_south_ray_from_center(self):
        """Test south ray from center square."""
        bb = make_uint64()
        result = get_south_ray(bb, 28)  # e4

        squares = get_squares_from_bitboard(result)
        expected = [20, 12, 4]  # e3, e2, e1
        assert sorted(squares) == sorted(expected)

    def test_south_ray_from_edge(self):
        """Test south ray from bottom edge (should be empty)."""
        bb = make_uint64()
        result = get_south_ray(bb, 4)  # e1

        squares = get_squares_from_bitboard(result)
        assert squares == []  # No squares south of bottom rank

    def test_east_ray_from_center(self):
        """Test east ray from center square."""
        bb = make_uint64()
        result = get_east_ray(bb, 28)  # e4

        squares = get_squares_from_bitboard(result)
        expected = [29, 30, 31]  # f4, g4, h4
        assert sorted(squares) == sorted(expected)

    def test_east_ray_from_edge(self):
        """Test east ray from right edge (should be empty)."""
        bb = make_uint64()
        result = get_east_ray(bb, 31)  # h4

        squares = get_squares_from_bitboard(result)
        assert squares == []  # No squares east of h-file

    def test_west_ray_from_center(self):
        """Test west ray from center square."""
        bb = make_uint64()
        result = get_west_ray(bb, 28)  # e4

        squares = get_squares_from_bitboard(result)
        expected = [27, 26, 25, 24]  # d4, c4, b4, a4
        assert sorted(squares) == sorted(expected)

    def test_west_ray_from_edge(self):
        """Test west ray from left edge (should be empty)."""
        bb = make_uint64()
        result = get_west_ray(bb, 24)  # a4

        squares = get_squares_from_bitboard(result)
        assert squares == []  # No squares west of a-file


class TestDiagonalRays:
    """Test diagonal ray generation (NE, NW, SE, SW)."""

    def test_northeast_ray_from_center(self):
        """Test northeast ray from center square."""
        bb = make_uint64()
        result = get_northeast_ray(bb, 28)  # e4

        squares = get_squares_from_bitboard(result)
        expected = [37, 46, 55]  # f5, g6, h7
        assert sorted(squares) == sorted(expected)

    def test_northeast_ray_from_corner(self):
        """Test northeast ray from bottom-left corner."""
        bb = make_uint64()
        result = get_northeast_ray(bb, 0)  # a1

        squares = get_squares_from_bitboard(result)
        expected = [9, 18, 27, 36, 45, 54, 63]  # b2, c3, d4, e5, f6, g7, h8
        assert sorted(squares) == sorted(expected)

    def test_northeast_ray_from_top_right(self):
        """Test northeast ray from top-right corner (should be empty)."""
        bb = make_uint64()
        result = get_northeast_ray(bb, 63)  # h8

        squares = get_squares_from_bitboard(result)
        assert squares == []  # No squares northeast of h8

    def test_northwest_ray_from_center(self):
        """Test northwest ray from center square."""
        bb = make_uint64()
        result = get_northwest_ray(bb, 28)  # e4

        squares = get_squares_from_bitboard(result)
        expected = [35, 42, 49, 56]  # d5, c6, b7, a8
        assert sorted(squares) == sorted(expected)

    def test_northwest_ray_from_corner(self):
        """Test northwest ray from bottom-right corner."""
        bb = make_uint64()
        result = get_northwest_ray(bb, 7)  # h1

        squares = get_squares_from_bitboard(result)
        expected = [14, 21, 28, 35, 42, 49, 56]  # g2, f3, e4, d5, c6, b7, a8
        assert sorted(squares) == sorted(expected)

    def test_southeast_ray_from_center(self):
        """Test southeast ray from center square."""
        bb = make_uint64()
        result = get_southeast_ray(bb, 28)  # e4

        squares = get_squares_from_bitboard(result)
        expected = [21, 14, 7]  # f3, g2, h1
        assert sorted(squares) == sorted(expected)

    def test_southeast_ray_from_corner(self):
        """Test southeast ray from top-left corner."""
        bb = make_uint64()
        result = get_southeast_ray(bb, 56)  # a8

        squares = get_squares_from_bitboard(result)
        expected = [49, 42, 35, 28, 21, 14, 7]  # b7, c6, d5, e4, f3, g2, h1
        assert sorted(squares) == sorted(expected)

    def test_southwest_ray_from_center(self):
        """Test southwest ray from center square."""
        bb = make_uint64()
        result = get_southwest_ray(bb, 28)  # e4

        squares = get_squares_from_bitboard(result)
        expected = [19, 10, 1]  # d3, c2, b1
        assert sorted(squares) == sorted(expected)

    def test_southwest_ray_from_corner(self):
        """Test southwest ray from top-right corner."""
        bb = make_uint64()
        result = get_southwest_ray(bb, 63)  # h8

        squares = get_squares_from_bitboard(result)
        expected = [54, 45, 36, 27, 18, 9, 0]  # g7, f6, e5, d4, c3, b2, a1
        assert sorted(squares) == sorted(expected)


class TestRayBoundaries:
    """Test ray generation at board boundaries."""

    def test_rays_from_corners(self):
        """Test all rays from corner squares."""
        corners = [0, 7, 56, 63]  # a1, h1, a8, h8

        for corner in corners:
            bb = make_uint64()

            # Test all ray directions
            north = get_north_ray(bb, corner)
            south = get_south_ray(bb, corner)
            east = get_east_ray(bb, corner)
            west = get_west_ray(bb, corner)
            ne = get_northeast_ray(bb, corner)
            nw = get_northwest_ray(bb, corner)
            se = get_southeast_ray(bb, corner)
            sw = get_southwest_ray(bb, corner)

            # At least some rays should be non-empty for corners
            # (except for the edges they're already on)
            total_squares = 0
            for ray in [north, south, east, west, ne, nw, se, sw]:
                total_squares += len(get_squares_from_bitboard(ray))

            # Each corner should have at least some valid ray directions
            assert total_squares > 0

    def test_rays_exclude_starting_square(self):
        """Test that rays don't include the starting square."""
        test_squares = [0, 28, 35, 63]  # Various positions

        for square in test_squares:
            bb = make_uint64()

            rays = [
                get_north_ray(bb, square),
                get_south_ray(bb, square),
                get_east_ray(bb, square),
                get_west_ray(bb, square),
                get_northeast_ray(bb, square),
                get_northwest_ray(bb, square),
                get_southeast_ray(bb, square),
                get_southwest_ray(bb, square),
            ]

            for ray in rays:
                squares = get_squares_from_bitboard(ray)
                assert square not in squares, f"Ray includes starting square {square}"


class TestRayConsistency:
    """Test consistency between different ray functions."""

    def test_opposite_rays_are_disjoint(self):
        """Test that opposite rays don't overlap."""
        test_square = 28  # e4
        bb = make_uint64()

        # Test straight rays
        north_squares = set(get_squares_from_bitboard(get_north_ray(bb, test_square)))
        south_squares = set(get_squares_from_bitboard(get_south_ray(bb, test_square)))
        east_squares = set(get_squares_from_bitboard(get_east_ray(bb, test_square)))
        west_squares = set(get_squares_from_bitboard(get_west_ray(bb, test_square)))

        assert north_squares.isdisjoint(south_squares)
        assert east_squares.isdisjoint(west_squares)

        # Test diagonal rays
        ne_squares = set(get_squares_from_bitboard(get_northeast_ray(bb, test_square)))
        sw_squares = set(get_squares_from_bitboard(get_southwest_ray(bb, test_square)))
        nw_squares = set(get_squares_from_bitboard(get_northwest_ray(bb, test_square)))
        se_squares = set(get_squares_from_bitboard(get_southeast_ray(bb, test_square)))

        assert ne_squares.isdisjoint(sw_squares)
        assert nw_squares.isdisjoint(se_squares)

    def test_ray_lengths_logical(self):
        """Test that ray lengths make sense given board geometry."""
        # From center square e4 (28)
        bb = make_uint64()

        north_len = len(get_squares_from_bitboard(get_north_ray(bb, 28)))
        south_len = len(get_squares_from_bitboard(get_south_ray(bb, 28)))
        east_len = len(get_squares_from_bitboard(get_east_ray(bb, 28)))
        west_len = len(get_squares_from_bitboard(get_west_ray(bb, 28)))

        # From e4: 4 squares north, 3 squares south, 3 squares east, 4 squares west
        assert north_len == 4  # e5, e6, e7, e8
        assert south_len == 3  # e3, e2, e1
        assert east_len == 3  # f4, g4, h4
        assert west_len == 4  # d4, c4, b4, a4


class TestRayEdgeCases:
    """Test edge cases and special conditions."""

    def test_rays_with_preset_bitboard(self):
        """Test ray generation when starting with non-empty bitboard."""
        # Start with a bitboard that has some bits set
        bb = np.uint64(0b1010101)

        # Ray generation should still work correctly
        # (the function should clear the starting square and add the ray)
        result = get_north_ray(bb, 28)

        # Should contain the original bits plus the north ray
        squares = get_squares_from_bitboard(result)

        # Original bits that aren't the starting square should still be there
        original_squares = get_squares_from_bitboard(bb)
        for sq in original_squares:
            if sq != 28:  # Starting square gets cleared
                assert sq in squares

        # North ray squares should be added
        expected_north = [36, 44, 52, 60]  # e5, e6, e7, e8
        for sq in expected_north:
            assert sq in squares

    def test_all_edge_squares(self):
        """Test ray generation from all edge squares."""
        # Test all squares on the edges of the board
        edge_squares = (
            list(range(0, 8))  # Rank 1
            + list(range(56, 64))  # Rank 8
            + [8, 16, 24, 32, 40, 48]  # A file (excluding corners)
            + [15, 23, 31, 39, 47, 55]  # H file (excluding corners)
        )

        bb = make_uint64()

        for square in edge_squares:
            # At least one ray direction should be blocked for edge squares
            rays = [
                get_north_ray(bb, square),
                get_south_ray(bb, square),
                get_east_ray(bb, square),
                get_west_ray(bb, square),
            ]

            # Count empty rays (blocked directions)
            empty_rays = sum(
                1 for ray in rays if len(get_squares_from_bitboard(ray)) == 0
            )

            # Edge squares should have at least one blocked direction
            assert (
                empty_rays >= 1
            ), f"Edge square {square} should have at least one blocked ray"

    def test_type_consistency(self):
        """Test that all ray functions return np.uint64."""
        bb = make_uint64()
        test_square = 28

        rays = [
            get_north_ray(bb, test_square),
            get_south_ray(bb, test_square),
            get_east_ray(bb, test_square),
            get_west_ray(bb, test_square),
            get_northeast_ray(bb, test_square),
            get_northwest_ray(bb, test_square),
            get_southeast_ray(bb, test_square),
            get_southwest_ray(bb, test_square),
        ]

        for ray in rays:
            assert isinstance(ray, np.uint64)
