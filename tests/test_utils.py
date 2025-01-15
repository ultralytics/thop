# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from thop import utils


class TestUtils:
    """Class for testing the clever_format function from the thop library for correct number formatting."""

    def test_clever_format_returns_formatted_number(self):
        """Tests that clever_format returns a string like '1.00B' for the given number and format pattern."""
        nums = 1
        format = "%.2f"
        clever_nums = utils.clever_format(nums, format)
        assert clever_nums == "1.00B"

    def test_clever_format_returns_formatted_numbers(self):
        """Verifies clever_format formats a list of numbers to strings, e.g., '[1, 2]' to '[1.00B, 2.00B]'."""
        nums = [1, 2]
        format = "%.2f"
        clever_nums = utils.clever_format(nums, format)
        assert clever_nums == ("1.00B", "2.00B")
