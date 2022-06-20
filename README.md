# Ten Bit Color Maps


## Features

- 1024 colors in each sequential colormap
- Perceptually uniform - each color is same perceptual distance from the previous color (before rounding)
- Raw color data rounded so they can be used in a byte tensor (see pytorch example below)
- All maps start at black

## Installation

```bash
pip install tbcm
```

## Usage

### With matplotlib

```python
from tbcm import tb_inferno, tb_oleron
```

then to use tb_inferno, do:

```python
import matplotlib.pyplot as plt
import numpy as np
x,y = np.meshgrid(np.linspace(-1,1,15),np.linspace(-1,1,15))
z = np.cos(x*np.pi)*np.sin(y*np.pi)

fig = plt.figure(figsize=(9,4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x,y,z,rstride=1,cstride=1,cmap=tb_inferno)
ax2 = fig.add_subplot(122)
cf = ax2.contourf(x,y,z,51,vmin=-1,vmax=1,cmap=tb_inferno)
cbar = fig.colorbar(cf)
```

result:

![matplotlib tb_inferno example](https://github.com/artiedins/tbcm/blob/main/images/mpl_tb_inferno.png)

and the same with tb_oleron is:

```python
fig = plt.figure(figsize=(9,4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x,y,z,rstride=1,cstride=1,cmap=tb_oleron)
ax2 = fig.add_subplot(122)
cf = ax2.contourf(x,y,z,51,vmin=-1,vmax=1,cmap=tb_oleron)
cbar = fig.colorbar(cf)
```

result:

![matplotlib tb_inferno example](https://github.com/artiedins/tbcm/blob/main/images/mlp_tb_oleron.png)


### With pytorch

```python
from tbcm import tb_inferno_data, tb_oleron_data

import torch
tb_inferno_cm = torch.tensor(tb_inferno_data, dtype=torch.uint8)
tb_oleron_cm = torch.tensor(tb_oleron_data, dtype=torch.uint8)
```

then to use tb_inferno, do:

```python
from PIL import Image

z = torch.cos(torch.linspace(-1, 1, 256).view(1, -1) * torch.pi) * torch.sin(
    torch.linspace(-1, 1, 256).view(-1, 1) * torch.pi)
z.add_(1).div_(2)

h,w = z.shape
idx = z.view(-1).mul(1023).round().clamp(0,1023).long()
img = tb_inferno_cm[idx].view(h,w,3)
Image.fromarray(img.numpy())
```

result:

![pytorch tb_inferno example](https://github.com/artiedins/tbcm/blob/main/images/pt_tb_inferno.png)

and the same with tb_oleron:

```python
h,w = z.shape
idx = z.view(-1).mul(1023).round().clamp(0,1023).long()
img = tb_oleron_cm[idx].view(h,w,3)
Image.fromarray(img.numpy())
```

result:

![pytorch tb_oleron example](https://github.com/artiedins/tbcm/blob/main/images/pt_tb_oleron.png)

