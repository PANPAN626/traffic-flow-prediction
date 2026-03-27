###调试allin
# import torch
# from model.layers import TemporalConvLayer
#
# torch.manual_seed(0)
#
# bs = 3
# c_in = 3
# c_out = 8
# ts = 12
# n_vertex = 5
# Kt = 3
#
# x = torch.randn(bs, c_in, ts, n_vertex)
#
# layer = TemporalConvLayer(
#     Kt=Kt,
#     c_in=c_in,
#     c_out=c_out,
#     n_vertex=n_vertex,
#     act_func='glu'
# )
#
# y = layer(x)
# print(y.shape)

###调试因果卷积
import torch
from model.layers import CausalConv2d
import torch
import torch.nn as nn

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        if enable_padding == True:
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0

        print("[init] kernel_size =", kernel_size)
        print("[init] dilation =", dilation)
        print("[init] padding =", self.__padding)

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, input):
        print("[forward] input.shape =", input.shape)
        result = super(CausalConv1d, self).forward(input)
        print("[forward] result.shape =", result.shape)

        if self.__padding != 0:
            output = result[:, :, :-self.__padding]
            print("[forward] output.shape =", output.shape)
            return output

        return result


if __name__ == "__main__":
    conv = CausalConv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        enable_padding=True
    )

    x = torch.randn(1, 1, 5)
    y = conv(x)
    print("final y.shape =", y.shape)