# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from thop.rnn_hooks import (
    count_gru,
    count_gru_cell,
    count_lstm,
    count_lstm_cell,
    count_rnn,
    count_rnn_cell,
    torch,
)
from thop.vision.basic_hooks import (
    count_adap_avgpool,
    count_avgpool,
    count_convNd,
    count_convtNd,
    count_linear,
    count_normalization,
    count_prelu,
    count_softmax,
    count_upsample,
    logging,
    nn,
    zero_ops,
)

from .utils import prRed
from .vision.calc_func import calculate_parameters

default_dtype = torch.float64

register_hooks = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convtNd,
    nn.ConvTranspose2d: count_convtNd,
    nn.ConvTranspose3d: count_convtNd,
    # the batch- and instance-norm families through the base they share, which count_normalization is
    # already annotated for. A lazy norm needs the base: LazyBatchNorm2d derives from _BatchNorm, not
    # from BatchNorm2d, so its concrete counterpart is a sibling and never appears in its MRO.
    nn.modules.batchnorm._NormBase: count_normalization,
    nn.LayerNorm: count_normalization,
    nn.PReLU: count_prelu,
    nn.Softmax: count_softmax,
    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: zero_ops,
    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: zero_ops,
    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,
    nn.RNNCell: count_rnn_cell,
    nn.GRUCell: count_gru_cell,
    nn.LSTMCell: count_lstm_cell,
    nn.RNN: count_rnn,
    nn.GRU: count_gru,
    nn.LSTM: count_lstm,
    nn.Sequential: zero_ops,
    nn.PixelShuffle: zero_ops,
}


def _resolve_rule(m_type, custom_ops, types_collection, verbose, report_missing):
    """Return the counting rule for a module type, or None when no ancestor of it carries one.

    The whole MRO is walked, most derived first, so a subclass of a supported type is counted by that type's rule
    instead of being skipped, and a lazy module resolves through the class it derives from. custom_ops is consulted
    ahead of the built-ins at every level, so a caller's own rule still wins.
    """
    first_seen = m_type not in types_collection
    for t in m_type.__mro__:
        if t in custom_ops:  # if defined in both op maps, custom_ops overwrites
            fn = custom_ops[t]
            if first_seen and verbose:
                print(f"[INFO] Customize rule {fn.__qualname__}() {m_type}.")
            return fn
        if t in register_hooks:
            fn = register_hooks[t]
            if first_seen and verbose:
                print(f"[INFO] Register {fn.__qualname__}() for {m_type}.")
            return fn
    if first_seen and report_missing:
        prRed(f"[WARN] Cannot find rule for {m_type}. Treat it as zero Macs.")
    return None


def _restore_modes(model, prev_training):
    """Put every module back into the training mode it was in before profiling.

    model.train() alone cannot do it: it recurses, so a tree holding more than one mode collapses into the root's. It
    runs first all the same, because a module that overrides train() — SAM's TinyViT attention drops a tensor it caches
    for eval — restores state of its own that no flag assignment reaches. The second pass repairs the modules that call
    flattened, through train() for the same reason.

    The third pass exists because the module graph is a DAG, not a tree: a module reached through two parents that
    disagree about its mode cannot be served by any order of recursive calls, and modules() yields it once, at the first
    parent, so the second parent's repair broadcasts over it unanswered. Assigning the flag is the only thing that
    always lands, and by then every override has already run.
    """
    model.train(prev_training[model])
    for m, was_training in prev_training.items():
        if m.training != was_training:
            m.train(was_training)
    for m, was_training in prev_training.items():
        m.training = was_training


def profile_origin(model, inputs, custom_ops=None, verbose=True, report_missing=False):
    """Profile a model with the legacy per-leaf-module traversal, returning total operations and parameters."""
    handler_collection = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    if report_missing:
        verbose = True

    def add_hooks(m):
        if list(m.children()):
            return

        if hasattr(m, "total_ops"):
            logging.warning(
                f".total_ops is already defined in {m!s}. Be careful, it might change your code's behavior."
            )

        m.register_buffer("total_ops", torch.zeros(1, dtype=default_dtype))

        m_type = type(m)
        fn = _resolve_rule(m_type, custom_ops, types_collection, verbose, report_missing)
        if fn is not None:
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)
        types_collection.add(m_type)

    # every module's own flag, not just the root's: model.eval() below recurses, so restoring the root alone
    # flattens a deliberately mixed tree, e.g. the frozen BatchNorm under a training parent that fine-tuning uses
    prev_training = {m: m.training for m in model.modules()}

    try:
        model.eval()
        model.apply(add_hooks)

        with torch.no_grad():
            model(*inputs)

        total_ops = 0
        for m in model.modules():
            if list(m.children()):  # skip for non-leaf module
                continue
            total_ops += m.total_ops

        total_ops = total_ops.item()
        total_params = float(calculate_parameters(model.parameters()))
    finally:  # no failure, at any stage, may leave behind the hooks or the buffers this call created
        for handler in handler_collection:
            handler.remove()
        for m in model.modules():
            if not list(m.children()):  # add_hooks only ever buffered leaves, so a container's is the caller's own
                m._buffers.pop("total_ops", None)
        _restore_modes(model, prev_training)  # last: it calls train(), which is the caller's code and may raise

    return total_ops, total_params


def profile(
    model: nn.Module,
    inputs,
    custom_ops=None,
    verbose=True,
    ret_layer_info=False,
    report_missing=False,
):
    """Profiles a PyTorch model, returning total operations, parameters, and optionally layer-wise details."""
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    if report_missing:
        # overwrite `verbose` option when enable report_missing
        verbose = True

    def add_hooks(m: nn.Module):
        """Registers a hook on a neural network module to track its total operations."""
        if m in handler_collection:  # model.apply() revisits modules shared by several parents, e.g. a common act
            return

        # a plain int attribute, not a float64 buffer: buffer reads go through nn.Module.__getattr__ and every
        # hook would allocate a tensor per call, which dominates profiling cost on module-heavy models.
        # Written straight into __dict__ (mirroring the teardown below) to skip nn.Module.__setattr__.
        m.__dict__["total_ops"] = 0

        m_type = type(m)
        fn = _resolve_rule(m_type, custom_ops, types_collection, verbose, report_missing)
        if fn is not None:
            # registered directly, as profile_origin does and as every release before 2.1.0 did, so a forward
            # pass spends no Python frame beyond the rule itself. A rule must therefore return None: its
            # return value is the module's output, per the forward-hook contract documented in the README
            handler_collection[m] = m.register_forward_hook(fn)
        types_collection.add(m_type)

    counted = set()

    def dfs_count(module: nn.Module, prefix="\t") -> (float, dict):
        """Recursively counts the total operations of the given PyTorch module and its submodules."""
        # float() rather than a bare read: a custom_ops rule may accumulate with a tensor, as the accumulator
        # in tests/test_custom_ops.py does, and callers are owed plain Python numbers
        total_ops = float(module.total_ops)
        ret_dict = {}
        for n, m in module.named_children():
            # every child is walked, whether or not it carries a rule of its own: a rule accounts for its
            # module's own arithmetic, not for its subtree's, so the two are added rather than one replacing
            # the other. This is what the root of the traversal has always done with its own total_ops above
            m_ops, next_dict = dfs_count(m, prefix=prefix + "\t")
            if ret_layer_info:  # the per-layer tree is the caller's opt-in, so it is not built when discarded
                ret_dict[n] = (m_ops, float(calculate_parameters(m.parameters())), next_dict)
            if m in counted:  # a module reached through several parents already accumulated all of its calls
                continue
            counted.add(m)
            total_ops += m_ops
        # print(prefix, module._get_name(), total_ops)
        return total_ops, ret_dict

    prev_training = {m: m.training for m in model.modules()}  # per module: model.eval() below recurses

    try:
        model.eval()
        model.apply(add_hooks)
        with torch.no_grad():
            model(*inputs)
        total_ops, ret_dict = dfs_count(model)
        # parameters belong to the module tree, not to the traced call, so they are read off the tree: a
        # per-module tally counts a weight shared by N modules N times, misses any module type without an
        # op-counting rule, and misses branches the forward pass skipped. parameters() dedups by tensor
        # identity and covers all three, and is what nn.Module reports for the same model.
        total_params = float(calculate_parameters(model.parameters()))
    finally:  # no failure, at any stage, may leave behind the hooks or the attribute this call created
        for handler in handler_collection.values():
            handler.remove()
        for m in model.modules():  # add_hooks ran on every module, so every module carries the temporary attribute
            m.__dict__.pop("total_ops", None)
        _restore_modes(model, prev_training)  # last: it calls train(), which is the caller's code and may raise

    if ret_layer_info:
        return total_ops, total_params, ret_dict
    return total_ops, total_params
