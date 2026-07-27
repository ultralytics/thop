# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections.abc import Iterable

COLOR_RED = "91m"
COLOR_GREEN = "92m"
COLOR_YELLOW = "93m"


def colorful_print(fn_print, color=COLOR_RED):
    """A decorator to print text in the specified terminal color by wrapping the given print function."""

    def actual_call(*args, **kwargs):
        print(f"\033[{color}", end="")
        fn_print(*args, **kwargs)
        print("\033[00m", end="")

    return actual_call


prRed = colorful_print(print, color=COLOR_RED)
prGreen = colorful_print(print, color=COLOR_GREEN)
prYellow = colorful_print(print, color=COLOR_YELLOW)


def clever_format(nums, format="%.2f"):
    """Formats numbers into human-readable strings with units (K for thousand, M for million, etc.)."""
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        for unit, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
            # abs() and >=, not the value against >: a negative satisfied no comparison and came out unscaled,
            # and an exact power of 1000 belongs to the unit it is one of rather than to the one below it
            if abs(num) >= scale:
                clever_nums.append(format % (num / scale) + unit)
                break
        else:
            clever_nums.append(format % num + "B")

    return clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)


if __name__ == "__main__":
    prRed("hello", "world")
    prGreen("hello", "world")
    prYellow("hello", "world")
