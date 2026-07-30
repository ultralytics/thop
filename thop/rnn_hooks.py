# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence


def _cell_evaluations(x):
    """Return how many cell evaluations a recurrent layer's input asks for."""
    if isinstance(x, PackedSequence):
        return int(x.batch_sizes.sum())  # the padded rectangle is the work packing exists to avoid
    return x.size(0) if x.dim() == 2 else x.size(0) * x.size(1)  # unbatched (L, H_in) carries no batch axis


def _count_rnn_cell(input_size, hidden_size, bias=True):
    """Calculate the total operations for an RNN cell given input size, hidden size, and optional bias."""
    total_ops = hidden_size * (input_size + hidden_size) + hidden_size
    if bias:
        total_ops += hidden_size * 2

    return total_ops


def count_rnn_cell(m: nn.RNNCell, x: torch.Tensor, y: torch.Tensor):
    """Counts the total RNN cell operations based on input tensor, hidden size, bias, and batch size."""
    total_ops = _count_rnn_cell(m.input_size, m.hidden_size, m.bias)

    # a cell takes one vector per batch entry, so a 1-dimensional input is a single unbatched sample
    total_ops *= x[0].size(0) if x[0].dim() == 2 else 1

    m.total_ops += int(total_ops)


def _count_gru_cell(input_size, hidden_size, bias=True):
    """Counts the total operations for a GRU cell based on input size, hidden size, and bias configuration."""
    total_ops = 0
    # r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
    # z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
    state_ops = (hidden_size + input_size) * hidden_size + hidden_size
    if bias:
        state_ops += hidden_size * 2
    total_ops += state_ops * 2

    # n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
    total_ops += (hidden_size + input_size) * hidden_size + hidden_size
    if bias:
        total_ops += hidden_size * 2
    # r hadamard : r * (~)
    total_ops += hidden_size

    # h' = (1 - z) * n + z * h
    # hadamard hadamard add
    total_ops += hidden_size * 3

    return total_ops


def count_gru_cell(m: nn.GRUCell, x: torch.Tensor, y: torch.Tensor):
    """Calculates and updates the total operations for a GRU cell in a mini-batch during inference."""
    total_ops = _count_gru_cell(m.input_size, m.hidden_size, m.bias)

    # a cell takes one vector per batch entry, so a 1-dimensional input is a single unbatched sample
    total_ops *= x[0].size(0) if x[0].dim() == 2 else 1

    m.total_ops += int(total_ops)


def _count_lstm_cell(input_size, hidden_size, bias=True, proj_size=0):
    """Counts LSTM cell operations during inference based on input size, hidden size, and bias configuration."""
    total_ops = 0
    real_hidden_size = proj_size or hidden_size  # what the recurrent matmuls read, torch's own name for it

    # i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
    # f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
    # o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
    # g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
    state_ops = (input_size + real_hidden_size) * hidden_size + hidden_size
    if bias:
        state_ops += hidden_size * 2
    total_ops += state_ops * 4

    # c' = f * c + i * g \\
    # hadamard hadamard add
    total_ops += hidden_size * 3

    # h' = W_{hr} (o * \tanh(c')) \\
    total_ops += hidden_size + hidden_size * proj_size

    return total_ops


def count_lstm_cell(m: nn.LSTMCell, x: torch.Tensor, y: torch.Tensor):
    """Counts and updates the total operations for an LSTM cell in a mini-batch during inference."""
    total_ops = _count_lstm_cell(m.input_size, m.hidden_size, m.bias)

    # a cell takes one vector per batch entry, so a 1-dimensional input is a single unbatched sample
    total_ops *= x[0].size(0) if x[0].dim() == 2 else 1

    m.total_ops += int(total_ops)


def count_rnn(m: nn.RNN, x, y):
    """Calculate and update the total number of operations for each RNN cell in a given batch."""
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    evaluations = _cell_evaluations(x[0])

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_rnn_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_rnn_cell(input_size, hidden_size, bias)

    for _ in range(num_layers - 1):
        total_ops += (
            _count_rnn_cell(hidden_size * 2, hidden_size, bias) * 2
            if m.bidirectional
            else _count_rnn_cell(hidden_size, hidden_size, bias)
        )
    total_ops *= evaluations

    m.total_ops += int(total_ops)


def count_gru(m: nn.GRU, x, y):
    """Calculates total operations for a GRU layer, updating the model's operation count based on batch size."""
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    evaluations = _cell_evaluations(x[0])

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_gru_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_gru_cell(input_size, hidden_size, bias)

    for _ in range(num_layers - 1):
        total_ops += (
            _count_gru_cell(hidden_size * 2, hidden_size, bias) * 2
            if m.bidirectional
            else _count_gru_cell(hidden_size, hidden_size, bias)
        )
    total_ops *= evaluations

    m.total_ops += int(total_ops)


def count_lstm(m: nn.LSTM, x, y):
    """Calculate total operations for LSTM layers, including bidirectional, updating model's total operations."""
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers
    proj_size = m.proj_size
    stacked_size = proj_size or hidden_size  # the width a projected layer hands the next one

    evaluations = _cell_evaluations(x[0])

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_lstm_cell(input_size, hidden_size, bias, proj_size) * 2
    else:
        total_ops += _count_lstm_cell(input_size, hidden_size, bias, proj_size)

    for _ in range(num_layers - 1):
        total_ops += (
            _count_lstm_cell(stacked_size * 2, hidden_size, bias, proj_size) * 2
            if m.bidirectional
            else _count_lstm_cell(stacked_size, hidden_size, bias, proj_size)
        )
    total_ops *= evaluations

    m.total_ops += int(total_ops)
