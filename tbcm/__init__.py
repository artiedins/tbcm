from .tb_inferno import tb_inferno, tb_inferno_data
from .tb_oleron import tb_oleron, tb_oleron_data
from .tb_sunrise import tb_sunrise, tb_sunrise_data

import torch
import numpy as np
from PIL import Image

tb_inferno_torch = torch.tensor(tb_inferno_data, dtype=torch.uint8)


def map_values(data, cm_range=(0.05, 0.98), invalid=None):
    if invalid is None:
        hist, bin_edges = torch.histogram(data.view(-1), bins=int(1e4))
    else:
        hist, bin_edges = torch.histogram(data[~invalid], bins=int(1e4))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cdf = hist.cumsum(dim=0)
    cdf = cdf / cdf[-1]
    out = torch.from_numpy(np.interp(data.numpy(), bin_centers.numpy(), cdf.numpy()))
    scale = cm_range[1] - cm_range[0]
    if invalid is None:
        out = out - out.min()
        out = out * scale / out.max()
    else:
        out = out - out[~invalid].min()
        out = out * scale / out[~invalid].max()
    out = out + cm_range[0]
    return out


def data_to_image(z, file_name, invalid=None):

    h, w = z.shape
    z = map_values(z, invalid=invalid).view(-1).mul(1023).round().clamp(0, 1023).long()
    z = tb_inferno_torch[z].view(h, w, 3)
    Image.fromarray(z.numpy()).save(file_name)
