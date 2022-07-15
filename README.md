Perceiver IO model implemented with Pytorch

[Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)

<div align="center">
  <img width="800" alt="perceiverio" src="https://user-images.githubusercontent.com/26805817/179248187-cb439069-6573-429e-bfff-70a67cbefff3.png">
</div>

## Quick Tour

```python
import torch

from perceiver.encoder import PerceiverEncoder
from perceiver.decoder import PerceiverDecoder
from perceiver.perceiver import PerceiverIO

>>> latent_dim = 512
>>> latent_num = 256
>>> input_dim = 128

>>> decoder_query_dim = 768

>>> encoder = PerceiverEncoder(
    input_dim=input_dim,
    latent_num=latent_num,
    latent_dim=latent_dim,
    cross_attn_heads=8,
    self_attn_heads=8,
    num_self_attn_per_block=6,
    num_self_attn_blocks=1
)

>>> decoder = PerceiverDecoder(
    q_dim=decoder_query_dim,
    latent_dim=latent_dim,
)

>>> perceiver = PerceiverIO(encoder=encoder, decoder=decoder)

>>> inputs = torch.randn(1, 12, input_dim)
>>> query = torch.randn(1, 23, decoder_query_dim)

>>> perceiver(inputs, query)
torch.Size([1, 23, 768])
tensor([[[ 0.9805, -0.6844,  1.3075,  ..., -0.9096,  0.6698, -0.5749],
         [ 0.7955, -1.1684,  1.8240,  ..., -0.5359,  2.0916,  0.7804],
         [ 0.6966, -2.2724,  0.6961,  ..., -0.0328,  0.9531,  0.7322],
         ...,
         [-0.5076,  1.0776, -0.2547,  ..., -0.2455,  0.1344,  1.3835],
         [-0.8452,  0.7084,  1.2860,  ..., -0.5268,  0.4482, -0.6365],
         [ 0.8611, -1.0659, -1.1967,  ...,  0.3491, -0.2891,  1.9208]]])
```

## Authors
* Mingu Kang - [Github](https://github.com/minqukanq)
