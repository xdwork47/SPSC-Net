from .unets_parts import *
from .transformer_partsR import TransformerDown,  TransformerDown_SPrune, TransformerDown_SPrune_Test, TransformerDown_Standard
from .multsc import Multiscale_spatial_channel_calibration
class SPSCNet(nn.Module):
    def __init__(self, down_block, n_channels, n_classes, imgsize, bilinear=True):
        super(SPSCNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.scale = 2  # 1 2 4

        self.inc = DoubleConv(n_channels, 64//self.scale)
        self.down1 = Down(64//self.scale, 128//self.scale)
        self.down2 = Down(128 // self.scale, 256 // self.scale)
        self.down3 = Down(256 // self.scale, 512 // self.scale)
        self.down4 = Down(512 // self.scale, 1024 // self.scale)
        factor = 2 if bilinear else 1
        self.trans5 = TransformerDown_SPrune(1024 // self.scale, 1024 // self.scale, imgsize //32, 4, heads=6, dim_head=128,
                                  patch_size=1,partition_size=7)
        self.conv5 = nn.Conv2d(1024//self.scale, 1024//self.scale//factor, kernel_size=1, padding=0, bias=False)

        self.up1 = Up(1024 // self.scale, 512 // factor // self.scale, bilinear)
        self.up2 = Up(512 // self.scale, 256 // factor // self.scale, bilinear)
        self.up3 = Up(256 // self.scale, 128 // factor // self.scale, bilinear)
        self.up4 = Up(128 // self.scale, 64 // self.scale, bilinear)
        self.outc = OutConv(64 // self.scale, n_classes)
        dims = [64//self.scale, 128//self.scale, 256//self.scale, 512//self.scale, 1024//self.scale]
        self.MSCR_1 = Multiscale_spatial_channel_calibration(attention_position=1, in_dim=dims, dim=128//self.scale)
        self.MSCR_2 = Multiscale_spatial_channel_calibration(attention_position=2, in_dim=dims, dim=256//self.scale)
        self.MSCR_3 = Multiscale_spatial_channel_calibration(attention_position=3, in_dim=dims, dim=512//self.scale)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)   #(512,28)
        x5 = self.down4(x4)   #(1024,14)

        encoder = [x1, x2, x3, x4, x5]
        x5, qkvs1, attns1 = self.trans5(x5)
        x5 = self.conv5(x5)

        x2 = self.MSCR_1(encoder)
        x3 = self.MSCR_2(encoder)
        x4 = self.MSCR_3(encoder)
        del encoder

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
def SPSC_Model(**kwargs):
    model = SPSCNet(TransformerDown_SPrune, **kwargs)
    return model
