import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple


kernel = '''
extern "C"
__global__ void recurrent_forget_mult(float *dst, const float *f, const float *x, int SEQ, int BATCH, int HIDDEN)
{
  /*
  Note: destination is assumed to be one timestep longer than f or x where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  if(hid >= HIDDEN || bid >= BATCH)
     return;
  //
  for (int ts = 0 + 1; ts < SEQ + 1; ts++) {
     // Good sanity check for debugging - only perform additions to a zeroed chunk of memory
     // Addition seems atomic or near atomic - you should get incorrect answers if doubling up via threads
     // Note: the index i needs to be offset by one as f[0] (f_t) is used for dst[1] (h_t) etc
     // To move timesteps, we step HIDDEN * BATCH
     // To move batches, we move HIDDEN
     // To move neurons, we move +- 1
     // Note: dst[dst_i] = ts * 100 + bid * 10 + hid; is useful for debugging
     int i           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_i       = (ts - 0) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_iminus1 = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     dst[dst_i]      = f[i] * x[i];
     dst[dst_i]      += (1 - f[i]) * dst[dst_iminus1];
  }
}
extern "C"
__global__ void bwd_recurrent_forget_mult(const float *h, const float *f, const float *x, const float *gh, float *gf, float *gx, float *ghinit, int SEQ, int BATCH, int HIDDEN)
{
  /*
  Note: h is assumed to be one timestep longer than f, x, gf, gx, or gh where dst[0] = h_{-1}
  This means dst array has a separate index than that of f or x
  */
  int hid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * blockDim.y + threadIdx.y;
  if(hid >= HIDDEN || bid >= BATCH)
     return;
  //
  double running_f = 0;
  for (int ts = SEQ - 1 + 1; ts >= 0 + 1; ts--) {
     int i           = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_i       = (ts - 0) * HIDDEN * BATCH + bid * HIDDEN + hid;
     int dst_iminus1 = (ts - 1) * HIDDEN * BATCH + bid * HIDDEN + hid;
     //
     running_f       += gh[dst_iminus1];
     // Gradient of X
     gx[i]           = f[i] * running_f;
     // Gradient of F
     gf[i]           = (x[i] - h[dst_iminus1]) * running_f;
     //
     // The line below is likely more numerically stable than (1 - f[i]) * running_f;
     running_f       = running_f - f[i] * running_f;
  }
  ghinit[bid * HIDDEN + hid] = running_f;
}
'''

###

class CPUForgetMult(torch.nn.Module):
    def __init__(self):
        super(CPUForgetMult, self).__init__()

    def forward(self, f, x, hidden_init=None):
        result = []
        ###
        forgets = f.split(1, dim=0)
        prev_h = hidden_init
        for i, h in enumerate((f * x).split(1, dim=0)):
            if prev_h is not None: h = h + (1 - forgets[i]) * prev_h
            # h is (1, batch, hidden) when it needs to be (batch_hidden)
            # Calling squeeze will result in badness if batch size is 1
            h = h.view(h.size()[1:])
            result.append(h)
            prev_h = h
        ###
        return torch.stack(result)


class GPUForgetMult(torch.autograd.Function):
    configured_gpus = {}
    ptx = None
    def __init__(self):
        super(GPUForgetMult, self).__init__()

    def compile(self):
        if self.ptx is None:
            program = Program(kernel.encode(), 'recurrent_forget_mult.cu'.encode())
            GPUForgetMult.ptx = program.compile()

        if torch.cuda.current_device() not in GPUForgetMult.configured_gpus:
            m = function.Module()
            m.load(bytes(self.ptx.encode()))

            self.forget_mult = m.get_function('recurrent_forget_mult')
            self.bwd_forget_mult = m.get_function('bwd_recurrent_forget_mult')

            Stream = namedtuple('Stream', ['ptr'])
            self.stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)

            GPUForgetMult.configured_gpus[torch.cuda.current_device()] = (self.forget_mult, self.bwd_forget_mult, self.stream)

        self.forget_mult, self.bwd_forget_mult, self.stream = GPUForgetMult.configured_gpus[torch.cuda.current_device()]

    def forward(self, f, x, hidden_init=None):
        self.compile()
        seq_size, batch_size, hidden_size = f.size()
        result = f.new(seq_size + 1, batch_size, hidden_size)
        # We only zero the result array (result[0]) if we don't set a hidden initial state
        # All other values (result[1:]) are overwritten by default
        if hidden_init is not None: result[0, :, :] = hidden_init
        else: result = result.zero_()
        ###
        grid_hidden_size = min(hidden_size, 512)
        grid = (math.ceil(hidden_size / grid_hidden_size), batch_size)
        self.forget_mult(grid=grid, block=(grid_hidden_size, 1), args=[result.data_ptr(), f.data_ptr(), x.data_ptr(), seq_size, batch_size, hidden_size], stream=self.stream)
        self.save_for_backward(f, x, hidden_init)
        self.result = result
        return result[1:, :, :]

    def backward(self, grad_h):
        self.compile()
        f, x, hidden_init = self.saved_tensors
        h = self.result
        ###
        seq_size, batch_size, hidden_size = f.size()
        # Zeroing is not necessary as these will be overwritten
        grad_f = f.new(*f.size())
        grad_x = f.new(*f.size())
        grad_h_init = f.new(batch_size, hidden_size)
        ###
        grid_hidden_size = min(hidden_size, 512)
        grid = (math.ceil(hidden_size / grid_hidden_size), batch_size)
        self.bwd_forget_mult(grid=grid, block=(grid_hidden_size, 1), args=[h.data_ptr(), f.data_ptr(), x.data_ptr(), grad_h.data_ptr(), grad_f.data_ptr(), grad_x.data_ptr(), grad_h_init.data_ptr(), seq_size, batch_size, hidden_size], stream=self.stream)
        ###
        if hidden_init is not None:
            return grad_f, grad_x, grad_h_init
        return grad_f, grad_x


class ForgetMult(torch.nn.Module):
    r"""ForgetMult computes a simple recurrent equation:
    h_t = f_t * x_t + (1 - f_t) * h_{t-1}
    This equation is equivalent to dynamic weighted averaging.
    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - F (seq_len, batch, input_size): tensor containing the forget gate values, assumed in range [0, 1].
        - hidden_init (batch, input_size): tensor containing the initial hidden state for the recurrence (h_{t-1}).
        - use_cuda: If True, use the fast element-wise CUDA kernel for recurrence. If False, uses naive for loop. Default: True.
    """

    def __init__(self):
        super(ForgetMult, self).__init__()

    def forward(self, f, x, hidden_init=None, use_cuda=True):
        # Use CUDA by default unless it's available
        use_cuda = use_cuda and torch.cuda.is_available()
        # Ensure the user is aware when ForgetMult is not GPU version as it's far faster
        if use_cuda: assert f.is_cuda and x.is_cuda, 'GPU ForgetMult with fast element-wise CUDA kernel requested but tensors not on GPU'
        ###
        # Avoiding 'RuntimeError: expected a Variable argument, but got NoneType' when hidden_init is None
        if hidden_init is None: return GPUForgetMult()(f, x) if use_cuda else CPUForgetMult()(f, x)
        return GPUForgetMult()(f, x, hidden_init) if use_cuda else CPUForgetMult()(f, x, hidden_init)


class QRNNLayer(nn.Module):
    r"""Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.
    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.
        use_cuda: If True, uses fast custom CUDA kernel. If False, uses naive for loop. Default: True.
    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (batch, hidden_size): tensor containing the initial hidden state for the QRNN.
    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size=None, save_prev_x=False, zoneout=0, window=10, output_gate=True,
                 use_cuda=True):
        super(QRNNLayer, self).__init__()

        # assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate
        self.use_cuda = use_cuda

        # One large matmul with concat is faster than N small matmuls and no concat
        self.linear = nn.Linear(self.window * self.input_size,
                                3 * self.hidden_size if self.output_gate else 2 * self.hidden_size)
        self.Conv1D = nn.Conv1d(self.input_size, 3 * self.hidden_size, kernel_size=self.window)

    def reset(self):
        # If you are saving the previous value of x, you should call this when starting with a new state
        self.prevX = None

    def forward(self, X, hidden=None):

        X = X.contiguous().permute(1, 0, 2)
        seq_len, batch_size, features_dim = X.size()

        if self.window > 1:
            K = torch.autograd.Variable(torch.zeros(self.window - 1, batch_size, features_dim),
                                        requires_grad=False).cuda()
            source = torch.cat((K, X), dim=0)

        source = source.contiguous().permute(1, 2, 0)
        Y = self.Conv1D(source)
        Y = Y.contiguous().permute(2, 0, 1)
        Y = Y.contiguous()

        # Convert the tensor back to (batch, seq_len, len([Z, F, O]) * hidden_size)
        if self.output_gate:
            # Y = Y.view(seq_len, batch_size, 3 * self.hidden_size)
            Z, F, O = Y.chunk(3, dim=2)
        else:
            # Y = Y.view(seq_len, batch_size, 2 * self.hidden_size)
            Z, F = Y.chunk(2, dim=2)
        ###
        Z = torch.nn.functional.tanh(Z)
        F = torch.nn.functional.sigmoid(F)

        # If zoneout is specified, we perform dropout on the forget gates in F
        # If an element of F is zero, that means the corresponding neuron keeps the old value
        if self.zoneout:
            if self.training:
                mask = Variable(F.data.new(*F.size()).bernoulli_(1 - self.zoneout), requires_grad=False)
                F = F * mask
            else:
                F *= 1 - self.zoneout

        # Ensure the memory is laid out as expected for the CUDA kernel
        # This is a null op if the tensor is already contiguous
        Z = Z.contiguous()
        F = F.contiguous()
        # The O gate doesn't need to be contiguous as it isn't used in the CUDA kernel

        # Forget Mult
        # For testing QRNN without ForgetMult CUDA kernel, C = Z * F may be useful
        C = ForgetMult()(F, Z, hidden, use_cuda=self.use_cuda)

        # Apply (potentially optional) output gate
        if self.output_gate:
            H = torch.nn.functional.sigmoid(O) * C
        else:
            H = C

        # In an optimal world we may want to backprop to x_{t-1} but ...
        if self.window > 1 and self.save_prev_x:
            self.prevX = Variable(X[-1:, :, :].data, requires_grad=False)

        H = H.contiguous().permute(1, 0, 2)

        return H, C[-1:, :, :]


class QRNN(torch.nn.Module):
    r"""Applies a multiple layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.
    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        num_layers: The number of QRNN layers to produce.
        layers: List of preconstructed QRNN layers to use for the QRNN module (optional).
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.
        use_cuda: If True, uses fast custom CUDA kernel. If False, uses naive for loop. Default: True.
    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (layers, batch, hidden_size): tensor containing the initial hidden state for the QRNN.
    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (layers, batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, layers=None, **kwargs):
        assert bidirectional == False, 'Bidirectional QRNN is not yet supported'
        assert batch_first == False, 'Batch first mode is not yet supported'
        assert bias == True, 'Removing underlying bias is not yet supported'

        super(QRNN, self).__init__()

        self.layers = torch.nn.ModuleList(
            layers if layers else [QRNNLayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in
                                   range(num_layers)])

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers) if layers else num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    def reset(self):
        r'''If your convolutional window is greater than 1, you must reset at the beginning of each new sequence'''
        [layer.reset() for layer in self.layers]

    def forward(self, input, hidden=None):
        next_hidden = []

        for i, layer in enumerate(self.layers):
            input, hn = layer(input, None if hidden is None else hidden[i])
            next_hidden.append(hn)

            if self.dropout != 0 and i < len(self.layers) - 1:
                input = torch.nn.functional.dropout(input, p=self.dropout, training=self.training, inplace=False)

        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers, *next_hidden[0].size()[-2:])

        return input, next_hidden