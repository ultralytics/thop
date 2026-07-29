# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import inspect

import pytest
import torch
from thop import profile, profile_origin
from torch import nn

# asked of torch rather than of thop, so this file states the platform boundary independently of the code under test
HOOK_TAKES_KWARGS = "with_kwargs" in inspect.signature(nn.Module.register_forward_hook).parameters
needs_kwargs = pytest.mark.skipif(
    not HOOK_TAKES_KWARGS, reason="torch<2.0 does not deliver keyword arguments to a forward hook"
)

# nn.Conv2d(3, 4, 3) on a 1x3x8x8 image: output 1x4x6x6, each output element costs one 3x3x3 kernel
CONV_3_4 = 4 * 6 * 6 * (3 * 3 * 3)
ENTRY_POINTS = [profile, profile_origin]


class Host(nn.Module):
    """Runs one inner module the way a test asks it to, so the inner call is the only thing under test."""

    def __init__(self, inner, call):
        """Hold the inner module and the callable that invokes it."""
        super().__init__()
        self.inner = inner
        self.call = call

    def forward(self, x):
        """Invoke the inner module through the stored callable."""
        return self.call(self.inner, x)


class TwoArguments(nn.Module):
    """A forward with a second, defaulted parameter, reachable positionally or by keyword."""

    def forward(self, a, mask=None):
        """Return the first argument untouched."""
        return a


class KeywordOnly(nn.Module):
    """A forward whose input cannot be passed positionally at all."""

    def forward(self, *, x):
        """Return the sole argument."""
        return x


class KeywordOnlyAfterOne(nn.Module):
    """A forward with an ordinary parameter and a keyword-only one behind it."""

    def forward(self, a, *, mask):
        """Return the first argument untouched."""
        return a


def _recorder(seen):
    """Return a counting rule that records the argument tuple it was handed and charges its input."""

    def record(m, x, y):
        """Record the tuple, then count the input the way a real rule reads it."""
        seen.append(x)
        m.total_ops += x[0].numel()

    return record


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_conv_called_by_keyword_is_counted_like_one_called_positionally(entry):
    """nn.Module.forward names its parameter "input", so calling a registered module by keyword is ordinary."""
    image = torch.zeros(1, 3, 8, 8)
    positional, _ = entry(Host(nn.Conv2d(3, 4, 3), lambda m, x: m(x)), inputs=(image,), verbose=False)
    assert positional == CONV_3_4  # the negative control: a call that already worked must not move
    if not HOOK_TAKES_KWARGS:
        pytest.skip("torch<2.0 does not deliver keyword arguments to a forward hook")
    keyword, _ = entry(Host(nn.Conv2d(3, 4, 3), lambda m, x: m(input=x)), inputs=(image,), verbose=False)
    assert keyword == positional


@needs_kwargs
@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize(
    "inner,call,expected",
    [
        (KeywordOnly, lambda m, x: m(x=x), 1),
        (KeywordOnlyAfterOne, lambda m, x: m(mask=x, a=x), 2),  # declared order reversed at the call site
    ],
)
def test_a_keyword_only_parameter_reaches_the_rule_in_declared_order(entry, inner, call, expected):
    """A keyword-only input cannot arrive positionally at all, so it is the case that most needs this."""
    seen = []
    a = torch.zeros(2)
    ops, _ = entry(Host(inner(), call), inputs=(a,), custom_ops={inner: _recorder(seen)}, verbose=False)
    assert [len(x) for x in seen] == [expected]
    assert ops == a.numel()  # x[0] is the parameter forward declares first, not the one passed first


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_mixed_call_reaches_the_rule_with_only_its_positional_arguments(entry):
    """Only an empty positional tuple is rewritten: what x holds elsewhere is the contract custom_ops is written to."""
    seen = []
    a = torch.zeros(2)
    entry(
        Host(TwoArguments(), lambda m, x: m(x, mask=x)),
        inputs=(a,),
        custom_ops={TwoArguments: _recorder(seen)},
        verbose=False,
    )
    assert [len(x) for x in seen] == [1]
