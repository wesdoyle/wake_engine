"""
Tests for the bitboard core module.

Tests all basic bitboard operations including creation, manipulation,
and querying functions.
"""

import numpy as np
import pytest

from wake.bitboard.core import (
    BOARD_SIZE,
    BOARD_SQUARES,
    make_uint64,
    get_bitboard_as_bytes,
    get_binary_string,
    get_squares_from_bitboard,
    bitscan_forward,
    bitscan_reverse,
    set_bit,
    clear_bit,
)


class TestConstants:
    """Test module constants."""

    def test_board_size(self):
        """Test that BOARD_SIZE is correct."""
        assert BOARD_SIZE == 8

    def test_board_squares(self):
        """Test that BOARD_SQUARES is correct."""
        assert BOARD_SQUARES == 64


class TestBitboardCreation:
    """Test bitboard creation and basic operations."""

    def test_make_uint64(self):
        """Test make_uint64 creates empty bitboard."""
        bb = make_uint64()
        assert isinstance(bb, np.uint64)
        assert bb == np.uint64(0)

    def test_get_bitboard_as_bytes(self):
        """Test bitboard to bytes conversion."""
        bb = make_uint64()
        bb_bytes = get_bitboard_as_bytes(bb)
        assert isinstance(bb_bytes, bytes)
        assert len(bb_bytes) == 8  # 64 bits = 8 bytes


class TestBinaryRepresentation:
    """Test binary string representation functions."""

    def test_get_binary_string_empty(self):
        """Test binary string of empty bitboard."""
        bb = make_uint64()
        binary_str = get_binary_string(bb)
        assert binary_str == "0" * 64
        assert len(binary_str) == 64

    def test_get_binary_string_single_bit(self):
        """Test binary string with single bit set."""
        bb = np.uint64(1)  # Bit 0 set
        binary_str = get_binary_string(bb)
        assert binary_str == "0" * 63 + "1"
        assert len(binary_str) == 64

    def test_get_binary_string_custom_size(self):
        """Test binary string with custom board size."""
        bb = np.uint64(15)  # 1111 in binary
        binary_str = get_binary_string(bb, 8)
        assert binary_str == "00001111"
        assert len(binary_str) == 8

    def test_get_squares_from_bitboard_empty(self):
        """Test getting squares from empty bitboard."""
        bb = make_uint64()
        squares = get_squares_from_bitboard(bb)
        assert squares == []

    def test_get_squares_from_bitboard_single(self):
        """Test getting squares from bitboard with single bit."""
        bb = np.uint64(1)  # Bit 0 set
        squares = get_squares_from_bitboard(bb)
        assert squares == [0]

    def test_get_squares_from_bitboard_multiple(self):
        """Test getting squares from bitboard with multiple bits."""
        bb = np.uint64(0b1010001)  # Bits 0, 4, 6 set
        squares = get_squares_from_bitboard(bb)
        assert sorted(squares) == [0, 4, 6]


class TestBitManipulation:
    """Test bit manipulation functions."""

    def test_set_bit_empty_board(self):
        """Test setting bit on empty bitboard."""
        bb = make_uint64()
        bb_result = set_bit(bb, 0)
        assert bb_result == np.uint64(1)
        assert bb == np.uint64(0)  # Original unchanged

    def test_set_bit_various_positions(self):
        """Test setting bits at various positions."""
        bb = make_uint64()

        # Test corner positions
        assert set_bit(bb, 0) == np.uint64(1)
        assert set_bit(bb, 7) == np.uint64(128)
        assert set_bit(bb, 56) == np.uint64(1) << 56
        assert set_bit(bb, 63) == np.uint64(1) << 63

    def test_set_bit_already_set(self):
        """Test setting a bit that's already set."""
        bb = np.uint64(1)  # Bit 0 already set
        bb_result = set_bit(bb, 0)
        assert bb_result == np.uint64(1)  # Should remain the same

    def test_clear_bit_single(self):
        """Test clearing a single bit."""
        bb = np.uint64(1)  # Bit 0 set
        bb_result = clear_bit(bb, 0)
        assert bb_result == np.uint64(0)

    def test_clear_bit_multiple(self):
        """Test clearing bit from bitboard with multiple bits."""
        bb = np.uint64(0b1010001)  # Bits 0, 4, 6 set
        bb_result = clear_bit(bb, 4)
        assert bb_result == np.uint64(0b1000001)  # Only bit 4 cleared

    def test_clear_bit_not_set(self):
        """Test clearing a bit that's not set."""
        bb = np.uint64(2)  # Only bit 1 set
        bb_result = clear_bit(bb, 0)
        assert bb_result == np.uint64(2)  # Should remain unchanged

    def test_bit_manipulation_chain(self):
        """Test chaining bit operations."""
        bb = make_uint64()
        bb = set_bit(bb, 0)
        bb = set_bit(bb, 5)
        bb = set_bit(bb, 10)
        bb = clear_bit(bb, 5)

        squares = get_squares_from_bitboard(bb)
        assert sorted(squares) == [0, 10]


class TestBitScanning:
    """Test bit scanning functions."""

    def test_bitscan_forward_single_bit(self):
        """Test forward bit scan with single bit."""
        bb = np.uint64(1)  # Bit 0 set
        result = bitscan_forward(bb)
        assert result == 1  # Note: function returns 1-based index

    def test_bitscan_forward_multiple_bits(self):
        """Test forward bit scan with multiple bits."""
        bb = np.uint64(0b1010000)  # Bits 4 and 6 set
        result = bitscan_forward(bb)
        assert result == 5  # Should find least significant bit (4 + 1)

    def test_bitscan_reverse_single_bit(self):
        """Test reverse bit scan with single bit."""
        bb = np.uint64(1) << 5  # Bit 5 set
        result = bitscan_reverse(bb)
        assert result == 5

    def test_bitscan_reverse_multiple_bits(self):
        """Test reverse bit scan with multiple bits."""
        bb = np.uint64(0b1010000)  # Bits 4 and 6 set
        result = bitscan_reverse(bb)
        assert result == 6  # Should find most significant bit

    def test_bitscan_reverse_empty_raises(self):
        """Test that reverse bit scan raises exception for empty bitboard."""
        bb = make_uint64()
        with pytest.raises(Exception, match="empty bitboard"):
            bitscan_reverse(bb)

    def test_bitscan_reverse_high_bit(self):
        """Test reverse bit scan with highest bit set."""
        bb = np.uint64(1) << 63  # Highest bit set
        result = bitscan_reverse(bb)
        assert result == 63


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_bits_set(self):
        """Test operations on bitboard with all bits set."""
        bb = np.uint64(0xFFFFFFFFFFFFFFFF)  # All bits set

        # Test that we get all 64 squares
        squares = get_squares_from_bitboard(bb)
        assert len(squares) == 64
        assert sorted(squares) == list(range(64))

        # Test bit scanning
        assert bitscan_forward(bb) == 1  # First bit (0-based becomes 1-based)
        assert bitscan_reverse(bb) == 63  # Last bit

    def test_boundary_positions(self):
        """Test operations at board boundaries."""
        boundary_positions = [0, 7, 8, 15, 48, 55, 56, 63]

        for pos in boundary_positions:
            bb = set_bit(make_uint64(), pos)
            squares = get_squares_from_bitboard(bb)
            assert squares == [pos]

            bb_cleared = clear_bit(bb, pos)
            assert bb_cleared == make_uint64()

    def test_type_consistency(self):
        """Test that all functions return consistent types."""
        bb = make_uint64()
        bb = set_bit(bb, 5)

        # All bitboard operations should return np.uint64
        assert isinstance(bb, np.uint64)
        assert isinstance(set_bit(bb, 10), np.uint64)
        assert isinstance(clear_bit(bb, 5), np.uint64)

        # Bit scanning should return int
        assert isinstance(bitscan_forward(bb), (int, np.integer))
        assert isinstance(bitscan_reverse(bb), (int, np.integer, np.uint64))


class TestPerformance:
    """Test performance characteristics (basic smoke tests)."""

    def test_operations_dont_mutate_original(self):
        """Test that bitboard operations don't mutate the original."""
        original = np.uint64(0b101010)

        # Test set_bit doesn't mutate
        result = set_bit(original, 0)
        assert original == np.uint64(0b101010)
        assert result != original

        # Test clear_bit doesn't mutate
        result = clear_bit(original, 1)
        assert original == np.uint64(0b101010)
        assert result != original


class TestBitScanningEdgeCases:
    """Test edge cases and boundary conditions for bit scanning functions."""

    def test_bitscan_forward_edge_cases(self):
        """Test bitscan_forward with edge case inputs."""
        # Test with zero bitboard
        result = bitscan_forward(np.uint64(0))
        assert result == -1  # Should return -1 for empty bitboard

        # Test with only highest bit set
        result = bitscan_forward(np.uint64(1) << np.uint64(63))
        assert result == 64  # 63 + 1 (1-based indexing)

        # Test with alternating bits
        alternating = np.uint64(0xAAAAAAAAAAAAAAAA)  # 1010...
        result = bitscan_forward(alternating)
        assert result == 2  # Lowest bit is at position 1, so 1+1=2 (1-based)

    def test_bitscan_reverse_edge_cases(self):
        """Test bitscan_reverse with edge case inputs."""
        # Test with single bit at position 0
        result = bitscan_reverse(np.uint64(1))
        assert result == 0

        # Test with bits in different byte boundaries
        # Test bit 7 (byte boundary)
        result = bitscan_reverse(np.uint64(1) << np.uint64(7))
        assert result == 7

        # Test bit 8 (crosses byte boundary)
        result = bitscan_reverse(np.uint64(1) << np.uint64(8))
        assert result == 8

        # Test bit 15 (2-byte boundary)
        result = bitscan_reverse(np.uint64(1) << np.uint64(15))
        assert result == 15

        # Test bit 16 (crosses 2-byte boundary)
        result = bitscan_reverse(np.uint64(1) << np.uint64(16))
        assert result == 16

        # Test bit 31 (4-byte boundary)
        result = bitscan_reverse(np.uint64(1) << np.uint64(31))
        assert result == 31

        # Test bit 32 (crosses 4-byte boundary)
        result = bitscan_reverse(np.uint64(1) << np.uint64(32))
        assert result == 32

    def test_bitscan_reverse_lookup_table_coverage(self):
        """Test bitscan_reverse lookup table function with different bit patterns."""
        # Test patterns that exercise the lookup_most_significant_1_bit function
        test_cases = [
            (np.uint64(0b00000001), 0),  # bit 0
            (np.uint64(0b00000010), 1),  # bit 1
            (np.uint64(0b00000100), 2),  # bit 2
            (np.uint64(0b00001000), 3),  # bit 3
            (np.uint64(0b00010000), 4),  # bit 4
            (np.uint64(0b00100000), 5),  # bit 5
            (np.uint64(0b01000000), 6),  # bit 6
            (np.uint64(0b10000000), 7),  # bit 7
            (np.uint64(0b11111111), 7),  # all bits set, should return highest
        ]

        for bit_pattern, expected_result in test_cases:
            result = bitscan_reverse(bit_pattern)
            # Convert numpy result to int for comparison
            assert int(result) == expected_result

    def test_bitscan_reverse_with_multiple_bits(self):
        """Test bitscan_reverse with multiple bits set."""
        # Test with bits at different positions
        multiple_bits = np.uint64(0b1010101010101010)
        result = bitscan_reverse(multiple_bits)
        # Should return the position of the highest bit (bit 15)
        assert result == 15

        # Test with pattern that has bits in upper half
        upper_bits = np.uint64(0xFF00000000000000)
        result = bitscan_reverse(upper_bits)
        assert result == 63  # Highest bit position

    def test_bitscan_functions_consistency(self):
        """Test that bitscan functions are consistent with each other."""
        # Test single bit patterns
        for bit_pos in [0, 1, 7, 8, 15, 16, 31, 32, 63]:
            single_bit = np.uint64(1) << np.uint64(bit_pos)

            # bitscan_forward should return bit_pos + 1 (1-based)
            forward_result = bitscan_forward(single_bit)
            assert forward_result == bit_pos + 1

            # bitscan_reverse should return bit_pos (0-based)
            reverse_result = bitscan_reverse(single_bit)
            assert reverse_result == bit_pos

    def test_bitscan_reverse_exception_handling(self):
        """Test that bitscan_reverse raises exception for empty bitboard."""
        with pytest.raises(Exception) as excinfo:
            bitscan_reverse(np.uint64(0))
        assert "empty bitboard" in str(excinfo.value).lower()


class TestBinaryRepresentationEdgeCases:
    """Test edge cases for binary representation functions."""

    def test_get_binary_string_edge_cases(self):
        """Test get_binary_string with edge case inputs."""
        # Test with maximum uint64 value
        max_val = np.uint64(0xFFFFFFFFFFFFFFFF)
        result = get_binary_string(max_val)
        assert result == "1" * 64
        assert len(result) == 64

        # Test with custom board size
        result = get_binary_string(np.uint64(0b1111), 8)
        assert result == "00001111"
        assert len(result) == 8

        # Test with custom board size - function pads with zeros, doesn't truncate
        result = get_binary_string(np.uint64(0b11), 4)
        assert result == "0011"  # Function pads to 4 characters
        assert len(result) == 4

        # Test that function doesn't truncate if binary is longer than board_squares
        result = get_binary_string(np.uint64(0b11111111), 4)
        assert result == "11111111"  # Function doesn't truncate, keeps original length
        assert len(result) == 8

    def test_get_squares_from_bitboard_edge_cases(self):
        """Test get_squares_from_bitboard with edge case inputs."""
        # Test with maximum uint64 value (all bits set)
        max_val = np.uint64(0xFFFFFFFFFFFFFFFF)
        result = get_squares_from_bitboard(max_val)
        assert len(result) == 64
        assert set(result) == set(range(64))

        # Test with alternating pattern
        alternating = np.uint64(0xAAAAAAAAAAAAAAAA)  # 1010...
        result = get_squares_from_bitboard(alternating)
        expected = [i for i in range(64) if i % 2 == 1]  # Odd positions
        assert set(result) == set(expected)

        # Test with single bit at highest position
        highest_bit = np.uint64(1) << np.uint64(63)
        result = get_squares_from_bitboard(highest_bit)
        assert result == [63]


class TestBitManipulationEdgeCases:
    """Test edge cases for bit manipulation functions."""

    def test_set_bit_edge_cases(self):
        """Test set_bit with edge case inputs."""
        # Test setting bit 0 and bit 63
        bb = make_uint64()

        # Set bit 0
        result = set_bit(bb, 0)
        assert result == np.uint64(1)

        # Set bit 63
        result = set_bit(bb, 63)
        assert result == np.uint64(1) << np.uint64(63)

        # Test setting all bits
        bb = make_uint64()
        for i in range(64):
            bb = set_bit(bb, i)

        # Should have all bits set
        assert bb == np.uint64(0xFFFFFFFFFFFFFFFF)

    def test_clear_bit_edge_cases(self):
        """Test clear_bit with edge case inputs."""
        # Start with all bits set
        bb = np.uint64(0xFFFFFFFFFFFFFFFF)

        # Clear bit 0
        result = clear_bit(bb, 0)
        expected = np.uint64(0xFFFFFFFFFFFFFFFE)
        assert result == expected

        # Clear bit 63
        result = clear_bit(bb, 63)
        expected = np.uint64(0x7FFFFFFFFFFFFFFF)
        assert result == expected

        # Test clearing all bits
        bb = np.uint64(0xFFFFFFFFFFFFFFFF)
        for i in range(64):
            bb = clear_bit(bb, i)

        # Should have no bits set
        assert bb == np.uint64(0)
