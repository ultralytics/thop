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
    count_relu,
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
    nn.BatchNorm1d: count_normalization,
    nn.BatchNorm2d: count_normalization,
    nn.BatchNorm3d: count_normalization,
    nn.LayerNorm: count_normalization,
    nn.InstanceNorm1d: count_normalization,
    nn.InstanceNorm2d: count_normalization,
    nn.InstanceNorm3d: count_normalization,
    nn.PReLU: count_prelu,
    nn.Softmax: count_softmax,
    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,
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
    nn.SyncBatchNorm: count_normalization,
}


def profile_origin(model, inputs, custom_ops=None, verbose=True, report_missing=False):
    """Profiles a PyTorch model's operations and parameters, applying either custom or default hooks."""
    handler_collection = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    if report_missing:
        verbose = True

    def add_hooks(m):
        if list(m.children()):
            return

        if hasattr(m, "total_ops") or hasattr(m, "total_params"):
            logging.warning(
                f"Either .total_ops or .total_params is already defined in {m!s}. "
                "Be careful, it might change your code's behavior."
            )

        m.register_buffer("total_ops", torch.zeros(1, dtype=default_dtype))
        m.register_buffer("total_params", torch.zeros(1, dtype=default_dtype))

        for p in m.parameters():
            m.total_params += torch.DoubleTensor([p.numel()])

        m_type = type(m)

        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print(f"[INFO] Customize rule {fn.__qualname__}() {m_type}.")
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print(f"[INFO] Register {fn.__qualname__}() for {m_type}.")
        else:
            if m_type not in types_collection and report_missing:
                prRed(f"[WARN] Cannot find rule for {m_type}. Treat it as zero Macs and zero Params.")

        if fn is not None:
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)
        types_collection.add(m_type)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if list(m.children()):  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    # remove temporal buffers
    for n, m in model.named_modules():
        if list(m.children()):
            continue
        if "total_ops" in m._buffers:
            m._buffers.pop("total_ops")
        if "total_params" in m._buffers:
            m._buffers.pop("total_params")

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
        """Registers hooks to a neural network module to track total operations and parameters."""
        if m in handler_collection:  # model.apply() revisits modules shared by several parents, e.g. a common act
            return

        # plain int attributes, not float64 buffers: buffer reads go through nn.Module.__getattr__ and every
        # hook would allocate a tensor per call, which dominates profiling cost on module-heavy models.
        # Written straight into __dict__ (mirroring the teardown below) to skip nn.Module.__setattr__.
        m.__dict__["total_ops"] = 0
        m.__dict__["total_params"] = 0

        m_type = type(m)

        fn = None
        if m_type in custom_ops:
            # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print(f"[INFO] Customize rule {fn.__qualname__}() {m_type}.")
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print(f"[INFO] Register {fn.__qualname__}() for {m_type}.")
        else:
            if m_type not in types_collection and report_missing:
                prRed(f"[WARN] Cannot find rule for {m_type}. Treat it as zero Macs and zero Params.")

        if fn is not None:
            # One hook, not two: every registered hook forces nn.Module.__call__ down its slow path, so the op
            # rule and the parameter tally share a single callback. Parameters cannot change during a forward
            # pass, so the count is taken once here rather than recomputed on each call.
            nparams = calculate_parameters(m.parameters(recurse=False))

            def counter(m, x, y, fn=fn, nparams=nparams):
                """Applies the module's op-counting rule and records its parameter count."""
                fn(m, x, y)
                m.__dict__["total_params"] = nparams

            handler_collection[m] = m.register_forward_hook(counter)
        types_collection.add(m_type)

    counted = set()

    def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
        """Recursively counts the total operations and parameters of the given PyTorch module and its submodules."""
        # float() rather than a bare read: a custom_ops rule may accumulate with a tensor, as the documented
        # `m.total_ops += torch.DoubleTensor([macs])` recipe does, and callers are owed plain Python numbers
        total_ops, total_params = float(module.total_ops), float(module.total_params)
        ret_dict = {}
        for n, m in module.named_children():
            next_dict = {}
            if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_ops, m_params = float(m.total_ops), float(m.total_params)
            else:
                m_ops, m_params, next_dict = dfs_count(m, prefix=prefix + "\t")
            ret_dict[n] = (m_ops, m_params, next_dict)
            if m in counted:  # a module reached through several parents already accumulated all of its calls
                continue
            counted.add(m)
            total_ops += m_ops
            total_params += m_params
        # print(prefix, module._get_name(), (total_ops, total_params))
        return total_ops, total_params, ret_dict

    prev_training_status = model.training

    try:
        model.eval()
        model.apply(add_hooks)
        with torch.no_grad():
            model(*inputs)
        total_ops, total_params, ret_dict = dfs_count(model)
    finally:  # no failure, at any stage, may leave hooks or buffers behind
        # reset model to original status
        model.train(prev_training_status)
        for handler in handler_collection.values():
            handler.remove()
        for m in model.modules():  # add_hooks ran on every module, so every module carries the temporary attributes
            m.__dict__.pop("total_ops", None)
            m.__dict__.pop("total_params", None)

    if ret_layer_info:
        return total_ops, total_params, ret_dict
    return total_ops, total_params
