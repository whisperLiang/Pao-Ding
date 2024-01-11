import torch.nn as nn
from enum import IntEnum
import typing
import torch
import torch.nn.functional as F

class DummyMHA(nn.Module):
    def __init__(self):
        super(DummyMHA, self).__init__()


class _CustomizedOp(nn.Module):
    def __init__(self, op_class):
        self.op_cls = op_class

    def __repr__(self):
        return "CustomizedOp({})".format(str(self.op_cls))
    

class _ConcatOp(nn.Module):
    def __init__(self, id):
        super(_ConcatOp, self).__init__()
        self.offsets = None
        self.concat_sizes = None
        self.id = id

    def __repr__(self):
        return "_ConcatOp_{}({})".format(self.id, self.offsets)
    
    def forward(self, input: typing.List[torch.Tensor]) -> torch.Tensor:
        # for densenet
        if len(input) == 1:
            return input
        else:
            return torch.cat(list(input), dim=1)


class _SplitOp(nn.Module):
    def __init__(self, id):
        super(_SplitOp, self).__init__()
        self.offsets = None
        self.split_sizes = None  
        self.id = id

    def __repr__(self):
        return "_SplitOp_{}({})".format(self.id,self.offsets)


class _UnbindOp(nn.Module):
    def __init__(self, id):
        super(_UnbindOp, self).__init__()
        self.offsets = None
        self.split_sizes = None  
        self.id = id

    def __repr__(self):
        return "_UnbindOp_{}({})".format(self.id,self.offsets)
    


class _ReshapeOp(nn.Module):
    def __init__(self, id):
        super(_ReshapeOp, self).__init__()
        self.id = id

    def __repr__(self):
        return "_Reshape_{}()".format(self.id)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.flatten(input, start_dim=1)
    
class _AddOp(nn.Module):
    def __init__(self, id):
        super(_AddOp, self).__init__()
        self.id = id

    def __repr__(self):
        return "_Add_{}()".format(self.id)
    
    def forward(self, input: typing.List[torch.Tensor]) -> torch.Tensor:
        return torch.add(*input)
    
class _PermuteOp(nn.Module):
    def __init__(self, id, grad_fn):
        super(_PermuteOp, self).__init__()
        self.grad_fn = grad_fn
        self.id = id

    def __repr__(self):
        return "_Permute_{}()".format(self.id)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dims = self.grad_fn._saved_dims
        return input.permute(dims)
    
    
class _ElementWiseOp(nn.Module):
    def __init__(self, id, grad_fn):
        super(_ElementWiseOp, self).__init__()
        self.ngrad_fn = grad_fn.name()
        self.id = id

    def __repr__(self):
        return "_ElementWiseOp_{}({})".format(self.id, self.ngrad_fn)


# Standard Modules
TORCH_CONV = nn.modules.conv._ConvNd
TORCH_BATCHNORM = nn.modules.batchnorm._BatchNorm
TORCH_LAYERNORM = nn.modules.normalization.LayerNorm
TORCH_GROUPNORM = nn.GroupNorm
TORCH_INSTANCENORM = nn.modules.instancenorm._InstanceNorm
TORCH_PRELU = nn.PReLU
TORCH_LINEAR = nn.Linear
TORCH_EMBED = nn.Embedding
TORCH_PARAMETER = nn.Parameter
TORCH_LSTM = nn.LSTM
TORCH_MAXPOOL2D = nn.modules.pooling.MaxPool2d
TORCH_ADAPTIVEAVGPOOL2D = nn.modules.AdaptiveAvgPool2d
try:
    TORCH_MHA = nn.MultiheadAttention
except:
    TORCH_MHA = DummyMHA  # for pytorch w/o MultiHeadAttention
TORCH_OTHERS = None


class OPTYPE(IntEnum):
    CONV = 0
    BN = 1
    LINEAR = 2
    PRELU = 3
    DEPTHWISE_CONV = 4
    CONCAT = 5  # torch.cat
    SPLIT = 6  # torch.split
    CUSTOMIZED = 7  # customized module
    ELEMENTWISE = 8  # element-wise add, sub, etc.
    LN = 9  # nn.LayerNorm
    EMBED = 10  # nn.Embedding
    PARAMETER = 11  # nn.Parameter
    MHA = 12
    LSTM = 13
    RESHAPE = 14
    GN = 15  # nn.GroupNorm
    IN = 16  # nn.InstanceNorm
    UNBIND = 17
    MAXPOOL2D = 18
    ADAPTIVEAVGPOOL2D = 19
    PERMUTE = 20



def module2type(module):
    if isinstance(module, TORCH_CONV):
        if module.groups == module.out_channels and module.out_channels > 1:
            return OPTYPE.DEPTHWISE_CONV
        else:
            return OPTYPE.CONV
    elif isinstance(module, TORCH_BATCHNORM):
        return OPTYPE.BN
    elif isinstance(module, TORCH_PRELU):
        return OPTYPE.PRELU
    elif isinstance(module, TORCH_LINEAR):
        return OPTYPE.LINEAR
    elif isinstance(module, _ConcatOp):
        return OPTYPE.CONCAT
    elif isinstance(module, _SplitOp):
        return OPTYPE.SPLIT
    elif isinstance(module, TORCH_LAYERNORM):
        return OPTYPE.LN
    elif isinstance(module, TORCH_EMBED):
        return OPTYPE.EMBED
    elif isinstance(module, _CustomizedOp):
        return OPTYPE.CUSTOMIZED
    elif isinstance(module, nn.Parameter):
        return OPTYPE.PARAMETER
    elif isinstance(module, TORCH_MHA):
        return OPTYPE.MHA
    elif isinstance(module, TORCH_LSTM):
        return OPTYPE.LSTM
    elif isinstance(module, TORCH_GROUPNORM):
        return OPTYPE.GN
    elif isinstance(module, TORCH_INSTANCENORM):
        return OPTYPE.IN
    elif isinstance(module, TORCH_MAXPOOL2D):
        return OPTYPE.MAXPOOL2D
    elif isinstance(module, TORCH_ADAPTIVEAVGPOOL2D):
        return OPTYPE.ADAPTIVEAVGPOOL2D
    elif isinstance(module, _ReshapeOp):
        return OPTYPE.RESHAPE
    elif isinstance(module, _UnbindOp):
        return OPTYPE.UNBIND
    elif isinstance(module, _PermuteOp):
        return OPTYPE.PERMUTE
    else:
        return OPTYPE.ELEMENTWISE


# def type2class(op_type):
#     if op_type == OPTYPE.CONV or op_type==OPTYPE.DEPTHWISE_CONV:
#         return TORCH_CONV
#     elif op_type == OPTYPE.BN:
#         return TORCH_BATCHNORM
#     # elif op_type == OPTYPE.RELU:
#     #     return TORCH_RELU
#     elif op_type == OPTYPE.PRELU:
#         return TORCH_PRELU
#     elif op_type == OPTYPE.LINEAR:
#         return TORCH_LINEAR
#     elif op_type == OPTYPE.CONCAT:
#         return _ConcatOp
#     elif op_type == OPTYPE.SPLIT:
#         return _SplitOp
#     elif op_type == OPTYPE.LN:
#         return TORCH_LAYERNORM
#     elif op_type == OPTYPE.EMBED:
#         return TORCH_EMBED
#     elif op_type == OPTYPE.CUSTOMIZED:
#         return _CustomizedOp
#     elif op_type == OPTYPE.PARAMETER:
#         return TORCH_PARAMETER
#     elif op_type == OPTYPE.MHA:
#         return TORCH_MHA
#     elif op_type == OPTYPE.GN:
#         return TORCH_GROUPNORM
#     elif op_type == OPTYPE.IN:
#         return TORCH_INSTANCENORM
#     elif op_type == OPTYPE.LSTM:
#         return TORCH_LSTM
#     elif op_type == OPTYPE.MAXPOOL2D:
#         return TORCH_MAXPOOL2D
#     elif op_type == OPTYPE.ADAPTIVEAVGPOOL2D:
#         return TORCH_ADAPTIVEAVGPOOL2D
#     elif OPTYPE == OPTYPE.RESHAPE:
#         return _ReshapeOp
#     elif OPTYPE == OPTYPE.UNBIND:
#         return _UnbindOp
#     else:
#         return _ElementWiseOp

