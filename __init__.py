import torch

from perceiver.encoder import PerceiverEncoder
from perceiver.decoder import PerceiverDecoder
from perceiver.perceiver import PerceiverIO


if __name__ == '__main__':
    latent_dim = 512
    latent_num = 256
    input_dim = 128

    decoder_query_dim = 768

    encoder = PerceiverEncoder(
        input_dim=input_dim,
        latent_num=latent_num,
        latent_dim=latent_dim,
        cross_attn_heads=8,
        self_attn_heads=8,
        num_self_attn_per_block=6,
        num_self_attn_blocks=1
    )

    decoder = PerceiverDecoder(
        q_dim=decoder_query_dim,
        latent_dim=latent_dim,
    )

    perceiver = PerceiverIO(encoder=encoder, decoder=decoder)

    inputs = torch.randn(1, 12, input_dim)
    query = torch.randn(1, 23, decoder_query_dim)

    # torch.Size([1, 23, 768])
    out = perceiver(inputs, query)
    print(out)
    print(out.shape)
    
