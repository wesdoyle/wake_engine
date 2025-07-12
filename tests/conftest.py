"""
Common pytest fixtures and configuration for bitboard tests.
"""

import pytest

from wake.bitboard import make_uint64, set_bit


@pytest.fixture
def empty_bitboard():
    """Fixture providing an empty bitboard."""
    return make_uint64()


@pytest.fixture
def single_bit_bitboard():
    """Fixture providing a bitboard with a single bit set at position 0."""
    return set_bit(make_uint64(), 0)


@pytest.fixture
def sample_squares():
    """Fixture providing common square indices for testing."""
    return {
        "a1": 0,
        "h1": 7,
        "a8": 56,
        "h8": 63,
        "e4": 28,
        "d4": 27,
        "e5": 36,
        "d5": 35,  # center squares
        "e1": 4,
        "e8": 60,  # king starting positions
    }


@pytest.fixture
def knight_squares():
    """Fixture providing knight square positions for testing."""
    return [1, 6, 8, 15, 17, 22]  # Various knight positions


@pytest.fixture
def edge_squares():
    """Fixture providing edge square positions."""
    return {
        "corners": [0, 7, 56, 63],
        "edges": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            15,
            16,
            23,
            24,
            31,
            32,
            39,
            40,
            47,
            48,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
        ],
    }
