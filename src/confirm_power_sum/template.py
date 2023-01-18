"""
sinc template: 8 x 1
Collect signal spread in multiple frequency bins
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Template(nn.Module):
    """
    Sinc template for subgrid frequency
    """
    def __init__(self, H: int, refine: int):
        super().__init__()

        w = create_template(H, refine)
        self.H = H  # kernel height
        self.refine = refine

        # kernel format: out_channels x in_channels x h x w
        w = w.reshape(refine, 1, self.H, 1)
        w = torch.from_numpy(w)  # refine x 1 x H x W
        self.register_buffer('w', w, persistent=False)

    def forward(self, x):
        """
        x (tensor[complex64]): 2 x H x W
        Returns: power of sinc weighted
          y (tensor[float32]): (H - H_template + 1, W)
        """
        assert x.ndim == 3
        nch, H_in, W_in = x.shape
        assert nch == 2

        # F.conv2d(x, w, stride)
        #   x: batch_size x in_channels x H x W
        #   w: out_channels x in_channels x h x w
        x = x.view(nch, 1, H_in, W_in)
        y = F.conv2d(x, self.w, stride=1)  # (2, refine, H - H_template + 1, W_in)
        y = y.real**2 + y.imag**2

        y = y.sum(dim=0)  # sum 2 channels

        refine, H2, W2 = y.shape
        y = y.transpose(0, 1).reshape(refine * H2, W2)
        return y


def create_template(H: int, refine: int):
    """
    Args:
      H (int): template height (=4)
      refint (int): frequency refining factor (=4)

    Returns:
      template (complex64): n_fdot x n_freq x H x 48
    """
    template = np.zeros((refine, H), dtype=np.complex64)

    f0s = np.arange(refine) / refine  # subgrid frequency in [0, 1)
    if H % 2 == 0:
        f_bin = np.arange(-(H // 2) + 1, H // 2 + 1)  # [-1, 0, 1, 2] for H=4
    else:
        f_bin = np.arange(-(H // 2), H // 2 + 1)  # (H, )  [-1, 0, 1] for H=3
    assert f_bin.shape[0] == H
    sign = 1 - 2 * (f_bin % 2)  # (-1)**f_bin

    for j, f in enumerate(f0s):
        # Amplitude = sinc(f - f_bin)
        w = sign * np.sinc(f - f_bin)  # H

        # Normalization: norm |w| = 1 conserve noise σ
        w_norm = np.sqrt(np.sum(w.real**2 + w.imag**2))
        w /= w_norm
        template[j] = w  # refine x H

    return template
