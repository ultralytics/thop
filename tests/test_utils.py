from thop import utils


class TestUtils:
    def test_clever_format_returns_formatted_number(self):
        """Tests that the clever_format function returns a formatted number string with a '1.00B' pattern."""
        nums = 1
        format = "%.2f"
        clever_nums = utils.clever_format(nums, format)
        assert clever_nums == "1.00B"

    def test_clever_format_returns_formatted_numbers(self):
        """Tests that the clever_format function correctly formats a list of numbers as strings with a '1.00B'
        pattern.
        """
        nums = [1, 2]
        format = "%.2f"
        clever_nums = utils.clever_format(nums, format)
        assert clever_nums == ("1.00B", "2.00B")
