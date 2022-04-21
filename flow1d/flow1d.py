import torch
import torch.nn as nn
import torch.nn.functional as F

from .extractor import BasicEncoder
from .attention import Attention1D
from .position import PositionEmbeddingSine
from .correlation import Correlation1D
from .update import BasicUpdateBlock
from utils.utils import coords_grid


class Model(nn.Module):
    def __init__(self,
                 downsample_factor=8,
                 feature_channels=256,
                 hidden_dim=128,
                 context_dim=128,
                 corr_radius=32,
                 mixed_precision=False,
                 **kwargs,
                 ):
        super(Model, self).__init__()

        self.downsample_factor = downsample_factor

        self.feature_channels = feature_channels

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.corr_radius = corr_radius

        self.mixed_precision = mixed_precision

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=feature_channels, norm_fn='instance',
                                 )

        self.cnet = BasicEncoder(output_dim=hidden_dim + context_dim, norm_fn='batch',
                                 )

        # 1D attention
        corr_channels = (2 * corr_radius + 1) * 2

        self.attn_x = Attention1D(feature_channels,
                                  y_attention=False,
                                  double_cross_attn=True,
                                  )
        self.attn_y = Attention1D(feature_channels,
                                  y_attention=True,
                                  double_cross_attn=True,
                                  )

        # Update block
        self.update_block = BasicUpdateBlock(corr_channels=corr_channels,
                                             hidden_dim=hidden_dim,
                                             context_dim=context_dim,
                                             downsample_factor=downsample_factor,
                                             )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, downsample=None):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        n, c, h, w = img.shape
        downsample_factor = self.downsample_factor if downsample is None else downsample
        coords0 = coords_grid(n, h // downsample_factor, w // downsample_factor).to(img.device)
        coords1 = coords_grid(n, h // downsample_factor, w // downsample_factor).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def learned_upflow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        n, _, h, w = flow.shape
        mask = mask.view(n, 1, 9, self.downsample_factor, self.downsample_factor, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.downsample_factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(n, 2, 9, 1, 1, h, w)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(n, 2, self.downsample_factor * h, self.downsample_factor * w)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False,
                ):
        """ Estimate optical flow between pair of frames """
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        # run the feature network
        feature1, feature2 = self.fnet([image1, image2])

        # Used for attention loss computation, store the attention matrix
        attn_x_list = []
        attn_y_list = []

        hdim = self.hidden_dim
        cdim = self.context_dim

        # position encoding
        pos_channels = self.feature_channels // 2
        pos_enc = PositionEmbeddingSine(pos_channels)

        position = pos_enc(feature1)  # [B, C, H, W]

        # 1D correlation
        feature2_x, attn_x = self.attn_x(feature1, feature2, position)
        corr_fn_y = Correlation1D(feature1, feature2_x,
                                  radius=self.corr_radius,
                                  x_correlation=False,
                                  )

        feature2_y, attn_y = self.attn_y(feature1, feature2, position)
        corr_fn_x = Correlation1D(feature1, feature2_y,
                                  radius=self.corr_radius,
                                  x_correlation=True,
                                  )

        # run the context network
        cnet = self.cnet(image1)  # list of feature pyramid, low scale to high scale

        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)  # 1/8 resolution or 1/4

        if flow_init is not None:  # flow_init is 1/8 resolution or 1/4
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()  # stop gradient

            corr_x = corr_fn_x(coords1)
            corr_y = corr_fn_y(coords1)
            corr = torch.cat((corr_x, corr_y), dim=1)  # [B, 2(2R+1), H, W]

            flow = coords1 - coords0

            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow,
                                                         upsample=not test_mode or itr == iters - 1,
                                                         )

            coords1 = coords1 + delta_flow

            if test_mode:
                # only upsample the last iteration
                if itr == iters - 1:
                    flow_up = self.learned_upflow(coords1 - coords0, up_mask)

                    return coords1 - coords0, flow_up
            else:
                # upsample predictions
                flow_up = self.learned_upflow(coords1 - coords0, up_mask)
                flow_predictions.append(flow_up)

        return flow_predictions, attn_x_list, attn_y_list, coords1 - coords0


def build_model(args):
    return Model(downsample_factor=args.downsample_factor,
                 feature_channels=args.feature_channels,
                 corr_radius=args.corr_radius,
                 hidden_dim=args.hidden_dim,
                 context_dim=args.context_dim,
                 mixed_precision=args.mixed_precision,
                 )
