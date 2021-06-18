import backbone
import head
import torch.nn as nn

class SRN(nn.Module):
    def __init__(self, option):
        super().__init__()
        self.backbone = backbone.ResNetFPN()
        self.head = head.SRNHead(out_channels=option.model.char_num,
                                 max_text_length=option.model.max_length,
                                 hidden_dims=option.model.hidden_dim,
                                 num_heads=option.model.num_heads,
                                 num_encoder_TUs=option.model.num_encoder_TUs,
                                 num_decoder_TUs=option.model.num_decoder_TUs)

    def forward(self, image, others):
        image = self.backbone(image)
        res = self.head(image, others)

        return res