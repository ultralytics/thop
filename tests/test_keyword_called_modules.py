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
IMAGE = torch.zeros(1, 3, 8, 8)


class CallsChild(nn.Module):
    """Wraps one convolution and decides how to hand it its input."""

    def __init__(self, keyword):
        """Build the child and record whether it is to be called by keyword."""
        super().__init__()
        self.c = nn.Conv2d(3, 4, 3)
        self.keyword = keyword

    def forward(self, x):
        """Call the child positionally or by keyword; nn.Module.forward names that parameter "input"."""
        return self.c(input=x) if self.keyword else self.c(x)


class CallsNorm(nn.Module):
    """Wraps one normalization layer, to show the fix sits at the hook and not inside the convolution rule."""

    def __init__(self, keyword):
        """Build the child and record whether it is to be called by keyword."""
        super().__init__()
        self.n = nn.LayerNorm(8)
        self.keyword = keyword

    def forward(self, x):
        """Call the child positionally or by keyword."""
        return self.n(input=x) if self.keyword else self.n(x)


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


class Variadic(nn.Module):
    """A forward whose parameter order no signature can report."""

    def forward(self, *args, **kwargs):
        """Return the sole argument, however it arrived."""
        return args[0] if args else next(iter(kwargs.values()))


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


def _recorder(seen, read_input=False):
    """Return a counting rule that records the argument tuple it was handed."""

    def record(m, x, y):
        """Record the tuple, and charge the input's element count when the test asks for it."""
        seen.append(x)
        m.total_ops += x[0].numel() if read_input else 0

    return record


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("model,image", [(CallsChild, IMAGE), (CallsNorm, torch.zeros(4, 8))])
def test_a_child_called_by_keyword_is_counted_like_one_called_positionally(entry, model, image):
    """The positional half is the negative control: the fix must not move a call that already worked."""
    positional, _ = entry(model(keyword=False), inputs=(image,), verbose=False)
    assert positional > 0
    if model is CallsChild:
        assert positional == CONV_3_4  # spelled out once, so the comparison below is not two unknowns
    if not HOOK_TAKES_KWARGS:
        pytest.skip("torch<2.0 does not deliver keyword arguments to a forward hook")
    keyword, _ = entry(model(keyword=True), inputs=(image,), verbose=False)
    assert keyword == positional


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_call_with_a_positional_argument_reaches_the_rule_untouched(entry):
    """The custom_ops contract: what x holds for a call that already worked does not change."""
    seen = []
    entry(
        Host(TwoArguments(), lambda m, x: m(x)),
        inputs=(torch.zeros(2),),
        custom_ops={TwoArguments: _recorder(seen)},
        verbose=False,
    )
    assert [len(x) for x in seen] == [1]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_mixed_call_reaches_the_rule_with_only_its_positional_arguments(entry):
    """The same control from the other side: keyword arguments beside a positional one are not rewritten either."""
    seen = []
    entry(
        Host(TwoArguments(), lambda m, x: m(x, mask=x)),
        inputs=(torch.zeros(2),),
        custom_ops={TwoArguments: _recorder(seen)},
        verbose=False,
    )
    assert [len(x) for x in seen] == [1]


@needs_kwargs
@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_an_all_keyword_call_reaches_the_rule_in_the_order_forward_declares(entry):
    """Not insertion order: the tuple a rule indexes has to mean what the signature says it means."""
    seen = []
    a, b = torch.zeros(2), torch.ones(2)
    entry(
        Host(TwoArguments(), lambda m, _: m(mask=b, a=a)),  # declared order reversed at the call site
        inputs=(a,),
        custom_ops={TwoArguments: _recorder(seen)},
        verbose=False,
    )
    assert [[id(v) for v in x] for x in seen] == [[id(a), id(b)]]


@needs_kwargs
@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize(
    "inner,call,expected",
    [
        (KeywordOnly, lambda m, x: m(x=x), 1),
        (KeywordOnlyAfterOne, lambda m, x: m(a=x, mask=x), 2),
    ],
)
def test_a_keyword_only_parameter_reaches_the_rule(entry, inner, call, expected):
    """A keyword-only input is the case least able to arrive positionally, so it is the one that most needs this."""
    seen = []
    a = torch.zeros(2)
    ops, _ = entry(
        Host(inner(), call), inputs=(a,), custom_ops={inner: _recorder(seen, read_input=True)}, verbose=False
    )
    assert [len(x) for x in seen] == [expected]
    assert ops == a.numel()


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_variadic_forward_called_by_keyword_stays_unorderable(entry):
    """The declined boundary: *args/**kwargs has no order to recover, and guessing one is worse than recovering none."""
    seen = []
    entry(
        Host(Variadic(), lambda m, x: m(anything=x)),
        inputs=(torch.zeros(2),),
        custom_ops={Variadic: _recorder(seen)},
        verbose=False,
    )
    assert seen == [()]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_rule_return_value_never_replaces_the_module_output(entry):
    """profile_origin registered the rule directly, so a rule that returned a value rewrote the forward pass."""
    seen = []

    def hijack(m, x, y):
        """A rule that returns a tensor, which a bare forward hook would substitute for the module output."""
        m.total_ops += 0
        return torch.full_like(y, 99.0)

    class Watcher(nn.Module):
        """Reads what its own child returned, the only place the substitution is observable."""

        def __init__(self):
            """Build the child convolution."""
            super().__init__()
            self.c = nn.Conv2d(3, 4, 3)

        def forward(self, x):
            """Record the largest value the child returned."""
            out = self.c(x)
            seen.append(float(out.max()))
            return out

    entry(Watcher(), inputs=(IMAGE,), custom_ops={nn.Conv2d: hijack}, verbose=False)
    assert seen and 99.0 not in seen
