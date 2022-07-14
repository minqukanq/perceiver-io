from typing import Optional

import torch
import torch.nn as nn

from .encoder import PerceiverEncoder
from .decoder import PerceiverDecoder


class PerceiverIO(nn.Module):

    def __init__(
        self,
        encoder: PerceiverEncoder,
        decoder: PerceiverDecoder
    ):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        inputs: Optional[torch.Tensor],
        query: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None
    ):

        latents = self.encoder(inputs, input_mask)

        outputs = self.decoder(
            x_q=query,
            latents=latents,
            query_mask=query_mask
        )
        return outputs