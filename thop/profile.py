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
    # the constant-padding and dropout families through the private base each shares, as the norm families
    # below need for the same reason: naming the two concrete classes left their 10 siblings rule-less, so
    # report_missing warned about layers that have nothing to count and taught the reader to ignore it
    nn.modules.padding._ConstantPadNd: zero_ops,  # padding does not involve any multiplication
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

if hasattr(nn, "RMSNorm"):  # torch>=2.4, and its elementwise_affine flag is one count_normalization already reads
    register_hooks[nn.RMSNorm] = count_normalization


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


def _restore_modes(prev_training):
    """Restore each module's training mode."""
    for m, was_training in prev_training.items():
        if m.training != was_training:
            m.train(was_training)
    for m, was_training in prev_training.items():
        m.training = was_training


def profile_origin(model, inputs, custom_ops=None, verbose=True, report_missing=False):
    """Profile a model with the legacy per-leaf-module traversal, returning total operations and parameters."""
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    if report_missing:
        verbose = True

    def add_hooks(m):
        if list(m.children()) or m in handler_collection:
            return

        if hasattr(m, "total_ops"):
            logging.warning(
                f".total_ops is already defined in {m!s}. Be careful, it might change your code's behavior."
            )

        m_type = type(m)
        fn = _resolve_rule(m_type, custom_ops, types_collection, verbose, report_missing)

        m.register_buffer("total_ops", torch.zeros(1, dtype=default_dtype))
        handler_collection[m] = None
        if fn is not None:
            handler_collection[m] = m.register_forward_hook(fn)
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

        # a plain int attribute, not a float64 buffer: buffer reads go through nn.Module.__getattr__ and every
        # hook would allocate a tensor per call, which dominates profiling cost on module-heavy models.
        # Written straight into __dict__ (mirroring the teardown below) to skip nn.Module.__setattr__.
        m.__dict__["total_ops"] = 0

        handler_collection[m] = None
        if fn is not None:

            def counter(m, x, y, fn=fn):
                """Apply the counting rule without allowing its return value to replace the module output."""
                fn(m, x, y)

            handler_collection[m] = m.register_forward_hook(counter)
        types_collection.add(m_type)

    counted = set()

    def dfs_count(module: nn.Module, prefix="\t") -> (float, dict):
        """Recursively counts the total operations of the given PyTorch module and its submodules."""
        # A custom rule may accumulate a tensor, and a module added during the forward has no temporary attribute.
        total_ops = float(module.__dict__.get("total_ops", 0))
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
            return dfs_count(model)

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
            required_samples = 2 if custom_ops or any(isinstance(m, fixed_ops) for m in model.modules()) else 1
            samples = []
            proxy_area = stride[0] * stride[1]
            if target_height % stride[0] == target_width % stride[1] == 0 and proxy_area < target_height * target_width:
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
        _restore_modes(prev_training)  # last: it calls train(), which is the caller's code and may raise

    if ret_layer_info:
        return total_ops, total_params, ret_dict
    return total_ops, total_params
