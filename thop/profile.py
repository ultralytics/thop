# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import inspect

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
    count_bilinear,
    count_convNd,
    count_convtNd,
    count_embedding,
    count_linear,
    count_multihead_attention,
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

# torch>=2.0
_HOOK_TAKES_KWARGS = "with_kwargs" in inspect.signature(nn.Module.register_forward_hook).parameters

register_hooks = {
    # the padding and dropout families through the private base each shares, as the norm families below
    # need for the same reason: naming the two concrete classes left their 10 siblings rule-less, so
    # report_missing warned about layers that have nothing to count and taught the reader to ignore it
    nn.modules.padding._ConstantPadNd: zero_ops,  # padding does not involve any multiplication
    nn.modules.padding._ReflectionPadNd: zero_ops,
    nn.modules.padding._ReplicationPadNd: zero_ops,
    nn.modules.dropout._DropoutNd: zero_ops,
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
    nn.GroupNorm: count_normalization,
    nn.PReLU: count_prelu,
    # The softmax family shares no base, so each member is registered explicitly.
    nn.Softmax: count_softmax,
    nn.LogSoftmax: count_softmax,
    nn.Softmin: count_softmax,
    nn.Softmax2d: count_softmax,
    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: zero_ops,
    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,
    nn.FractionalMaxPool2d: zero_ops,  # a max pool that picks its windows randomly still only selects
    nn.FractionalMaxPool3d: zero_ops,
    nn.modules.pooling._MaxUnpoolNd: zero_ops,  # scattering into a larger zero tensor adds nothing
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Bilinear: count_bilinear,
    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,
    nn.RNNCell: count_rnn_cell,
    nn.GRUCell: count_gru_cell,
    nn.LSTMCell: count_lstm_cell,
    nn.RNN: count_rnn,
    nn.GRU: count_gru,
    nn.LSTM: count_lstm,
    # containers hold children and compute nothing themselves; they share no base but nn.Module.
    nn.Sequential: zero_ops,
    nn.ModuleList: zero_ops,
    nn.ModuleDict: zero_ops,
    nn.ParameterList: zero_ops,
    nn.ParameterDict: zero_ops,
    # layers that only move elements around: a view, a re-index or a permutation reads and writes each
    # element once and multiplies nothing. The elementwise activations that also reach no rule are left
    # warned about on purpose, because registering one asserts it is free and SiLU, Mish, GELU, GLU and
    # Hardswish each multiply per element, so they want formulas rather than a blanket entry. nn.Fold is
    # out because it sums OVERLAPPING blocks, and an addition per input over a window is the work
    # count_adap_avgpool charges rather than something a view does.
    nn.Identity: zero_ops,
    nn.Flatten: zero_ops,
    nn.Unflatten: zero_ops,
    nn.Unfold: zero_ops,
    nn.PixelShuffle: zero_ops,
    nn.PixelUnshuffle: zero_ops,
    nn.ChannelShuffle: zero_ops,
    # a gather, so it belongs with the block above, but max_norm keeps it out of a blanket zero: that rescale is
    # real work and count_embedding warns rather than assert it away, the way count_upsample does for a mode it has
    # no cost for. nn.EmbeddingBag stays unregistered, since it really does reduce each bag and how many adds that
    # takes is not a function of shape either: padding_idx entries drop out and repeated offsets make empty bags.
    nn.Embedding: count_embedding,
}

if _HOOK_TAKES_KWARGS:
    register_hooks[nn.MultiheadAttention] = count_multihead_attention

if hasattr(nn, "RMSNorm"):  # torch>=2.4, and its elementwise_affine flag is one count_normalization already reads
    register_hooks[nn.RMSNorm] = count_normalization

if hasattr(nn.modules.padding, "_CircularPadNd"):  # torch>=2.1, which is where the CircularPad classes start
    register_hooks[nn.modules.padding._CircularPadNd] = zero_ops


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
                print(f"[INFO] Customize rule {getattr(fn, '__qualname__', fn)}() {m_type}.")
            return fn
        if t in register_hooks:
            fn = register_hooks[t]
            if first_seen and verbose:
                print(f"[INFO] Register {getattr(fn, '__qualname__', fn)}() for {m_type}.")
            return fn
    if first_seen and report_missing:
        prRed(f"[WARN] Cannot find rule for {m_type}. Treat it as zero Macs.")
    return None


def _register_counter(m, fn, keep_result):
    """Register a counting hook that receives positional and keyword arguments."""
    if _HOOK_TAKES_KWARGS:

        def counter(m, args, kwargs, y, fn=fn):
            """Apply the counting rule to the arguments the module was called with."""
            result = fn(m, _positional(m, args, kwargs) if kwargs else args, y)
            return result if keep_result else None

        return m.register_forward_hook(counter, with_kwargs=True)

    def counter(m, args, y, fn=fn):
        """Apply the counting rule; this torch delivers no keyword arguments for it to be given."""
        result = fn(m, args, y)
        return result if keep_result else None

    return m.register_forward_hook(counter)


def _positional(m, args, kwargs):
    """Return supplied arguments in the order m.forward declares them."""
    try:
        bound = inspect.signature(m.forward).bind(*args, **kwargs)
    except (TypeError, ValueError):
        return ()
    values = []
    for name, value in bound.arguments.items():
        kind = bound.signature.parameters[name].kind
        if kind == inspect.Parameter.VAR_POSITIONAL:
            values.extend(value)
        elif kind != inspect.Parameter.VAR_KEYWORD:
            values.append(value)
    return tuple(values)


def _parents_first(roots):
    """List reachable modules with every parent before its children."""
    order, seen = [], set()

    def visit(m):
        seen.add(m)
        for c in m.children():
            if c not in seen:
                visit(c)
        order.append(m)

    for root in roots:
        if root not in seen:
            visit(root)
    order.reverse()
    return order


def _displace_total_ops(m):
    """Temporarily remove a caller-owned `total_ops` buffer or attribute; reject a name thop cannot borrow."""
    kind = None
    if any(hasattr(vars(c).get("total_ops"), "__set__") for c in type(m).__mro__):
        kind = "data descriptor"  # no per-instance value to displace, and the hook's writes never reach the total
    elif "total_ops" in m._parameters:
        kind = "parameter"  # displacing it would come back as a smaller parameter count
    elif "total_ops" in m._modules:
        kind = "child module"  # displacing it would come back as a dropped subtree
    if kind:
        raise TypeError(f"{m!s} owns total_ops as a {kind}; thop needs that name to profile it.")
    displaced = [(store, store["total_ops"]) for store in (m._buffers, m.__dict__) if "total_ops" in store]
    if displaced:
        logging.warning(f"{m!s} already has a .total_ops; it is shadowed while profiling and restored after.")
        for store, _ in displaced:
            del store["total_ops"]
    return displaced


def _restore_modes(prev_training):
    """Restore recorded modes without changing modules attached during forward."""
    order = _parents_first(prev_training)
    modes = {m: prev_training.get(m, m.training) for m in order}  # read before the first train() call moves any
    for m, was_training in modes.items():
        if m.training != was_training:
            m.train(was_training)
    for m, was_training in modes.items():
        m.training = was_training


def profile_origin(model, inputs, custom_ops=None, verbose=True, report_missing=False):
    """Profile a model with the legacy per-leaf-module traversal, returning total operations and parameters."""
    handler_collection = {}
    displaced_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    if report_missing:
        verbose = True

    def add_hooks(m):
        m_type = type(m)
        has_own_rule = any(t in custom_ops or t in register_hooks for t in m_type.__mro__)
        if (list(m.children()) and not has_own_rule) or m in handler_collection:
            return

        fn = _resolve_rule(m_type, custom_ops, types_collection, verbose, report_missing)

        displaced = _displace_total_ops(m)
        if displaced:
            displaced_collection[m] = displaced
        # register_buffer discards this flag; restore it without requiring the newer persistent= argument.
        non_persistent = "total_ops" in m._non_persistent_buffers_set
        m.register_buffer("total_ops", torch.zeros(1, dtype=default_dtype))
        if non_persistent:
            m._non_persistent_buffers_set.add("total_ops")
        handler_collection[m] = None
        if fn is not None and fn is not zero_ops:
            handler_collection[m] = _register_counter(m, fn, keep_result=True)  # as profile_origin always has
        types_collection.add(m_type)

    prev_training = {m: m.training for m in model.modules()}

    try:
        model.eval()
        model.apply(add_hooks)

        with torch.no_grad():
            model(*inputs)

        total_ops = 0
        for m in handler_collection:
            total_ops += m.total_ops

        total_ops = total_ops.item()
        total_params = float(calculate_parameters(model.parameters()))
    finally:  # no failure, at any stage, may leave behind the hooks or the buffers this call created
        for handler in filter(None, handler_collection.values()):
            handler.remove()
        for m in handler_collection:
            m._buffers.pop("total_ops", None)
        for m, displaced in displaced_collection.items():
            for store, value in displaced:
                store["total_ops"] = value
        _restore_modes(prev_training)  # last: it calls train(), which is the caller's code and may raise

    return total_ops, total_params


def profile(
    model: nn.Module,
    inputs,
    custom_ops=None,
    verbose=True,
    ret_layer_info=False,
    report_missing=False,
    stride=None,
):
    """Profiles a PyTorch model, optionally estimating target-image MACs from smaller stride-aligned inputs."""
    handler_collection = {}
    displaced_collection = {}
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

        m_type = type(m)
        fn = _resolve_rule(m_type, custom_ops, types_collection, verbose, report_missing)

        displaced = _displace_total_ops(m)
        if displaced:
            displaced_collection[m] = displaced
        # a plain int attribute, not a float64 buffer: buffer reads go through nn.Module.__getattr__ and every
        # hook would allocate a tensor per call, which dominates profiling cost on module-heavy models.
        # Written straight into __dict__ (mirroring the teardown below) to skip nn.Module.__setattr__.
        m.__dict__["total_ops"] = 0

        handler_collection[m] = None
        if fn is not None and fn is not zero_ops:
            handler_collection[m] = _register_counter(m, fn, keep_result=False)  # never the module output
        types_collection.add(m_type)

    counted = set()

    def dfs_count(module: nn.Module, prefix="\t") -> (float, dict):
        """Build layer details from the module tree left after forward."""
        total_ops = float(module.__dict__.get("total_ops", 0)) if module in handler_collection else 0.0
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

        def run(input_values):
            """Profile one set of inputs while reusing the registered hooks."""
            counted.clear()
            for m in handler_collection:
                m.__dict__["total_ops"] = 0
            with torch.no_grad():
                model(*input_values)
            # Count the executed hook record; build the surviving tree only when requested.
            total = sum(float(m.__dict__.get("total_ops", 0)) for m in handler_collection)
            return total, dfs_count(model)[1] if ret_layer_info else {}

        if stride is None or ret_layer_info:
            total_ops, ret_dict = run(inputs)
        else:
            if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor) or inputs[0].ndim != 4:
                raise ValueError("stride requires inputs to contain one BCHW image tensor")
            stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
            if len(stride) != 2 or min(stride) < 1:
                raise ValueError("stride must be a positive integer or a pair of positive integers")

            image = inputs[0]
            target_height, target_width = image.shape[-2:]
            fixed_ops = (
                nn.Linear,
                nn.AdaptiveAvgPool1d,
                nn.AdaptiveAvgPool2d,
                nn.AdaptiveAvgPool3d,
                nn.RNNBase,
                nn.RNNCell,
                nn.GRUCell,
                nn.LSTMCell,
            )
            required_samples = 2 if any(isinstance(m, fixed_ops) for m in model.modules()) else 1
            samples = []
            proxy_area = stride[0] * stride[1]
            if (
                # two small samples can only fit a cost that is affine in image area. Attention is quadratic in it,
                # and a caller's own rule may be anything at all, so a model carrying either is measured directly.
                not any(
                    isinstance(m, nn.MultiheadAttention) or any(t in custom_ops for t in type(m).__mro__)
                    for m in model.modules()
                )
                and target_height % stride[0] == target_width % stride[1] == 0
                and proxy_area < target_height * target_width
            ):
                try:
                    for width in (stride[1], stride[1] * 2)[:required_samples]:
                        ops, _ = run((image.new_empty((*image.shape[:-2], stride[0], width)),))
                        samples.append((stride[0] * width, ops))
                except Exception:
                    samples.clear()

            if len(samples) == 2:
                (area1, ops1), (area2, ops2) = samples
                slope = (ops2 - ops1) / (area2 - area1)
                total_ops = slope * target_height * target_width + ops1 - slope * area1
                ret_dict = {}
            elif len(samples) == 1 and required_samples == 1:
                area, ops = samples[0]
                total_ops = ops * target_height * target_width / area
                ret_dict = {}
            else:
                total_ops, ret_dict = run(inputs)
        # Parameters belong to the module tree, not to the traced call. parameters() deduplicates shared tensors
        # and includes unsupported or unexecuted modules.
        total_params = float(calculate_parameters(model.parameters()))
    finally:  # no failure, at any stage, may leave behind hooks or temporary attributes
        for handler in filter(None, handler_collection.values()):
            handler.remove()
        for m in handler_collection:
            m.__dict__.pop("total_ops", None)
        for m, displaced in displaced_collection.items():
            for store, value in displaced:
                store["total_ops"] = value
        _restore_modes(prev_training)  # last: it calls train(), which is the caller's code and may raise

    if ret_layer_info:
        return total_ops, total_params, ret_dict
    return total_ops, total_params
