
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


ws = 512


class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        # image1
        resnet_im1 = models.resnet101(pretrained=True)
        self.conv1_1 = resnet_im1.conv1
        self.bn1_1 = resnet_im1.bn1
        self.relu_1 = resnet_im1.relu
        self.maxpool_1 = resnet_im1.maxpool

        self.res2_1 = resnet_im1.layer1
        self.res3_1 = resnet_im1.layer2
        self.res4_1 = resnet_im1.layer3
        self.res5_1 = resnet_im1.layer4

        # image2
        resnet_im2 = models.resnet101(pretrained=True)
        self.conv1_2 = resnet_im2.conv1
        self.bn1_2 = resnet_im2.bn1
        self.relu_2 = resnet_im2.relu
        self.maxpool_2 = resnet_im2.maxpool

        self.res2_2 = resnet_im2.layer1
        self.res3_2 = resnet_im2.layer2
        self.res4_2 = resnet_im2.layer3
        self.res5_2 = resnet_im2.layer4

        # flow
        resnet_fl = models.resnet101(pretrained=True)
        self.conv1_3 = resnet_fl.conv1
        self.bn1_3 = resnet_fl.bn1
        self.relu_3 = resnet_fl.relu
        self.maxpool_3 = resnet_fl.maxpool

        self.res2_3 = resnet_fl.layer1
        self.res3_3 = resnet_fl.layer2
        self.res4_3 = resnet_fl.layer3
        self.res5_3 = resnet_fl.layer4

        # GAC
        self.gated_res2 = GAC(256*2)
        self.gated_res3 = GAC(512*2)
        self.gated_res4 = GAC(1024*2)
        self.gated_res5 = GAC(2048*2)

        # parallel co-attention
        self.pca_res2 = ParallelCA(channel=128)
        self.pca_res3 = ParallelCA(channel=512)
        self.pca_res4 = ParallelCA(channel=1024)
        self.pca_res5 = ParallelCA(channel=2048)

        # cross co-attention
        self.cca_res2 = CrossCoAttention(channel=256)
        self.cca_res3 = CrossCoAttention(channel=512)
        self.cca_res4 = CrossCoAttention(channel=1024)
        self.cca_res5 = CrossCoAttention(channel=2048)

    # f1: image1 f2: image2 f3: flow
    def forward_res2(self, f1, f2, f3):
        x1 = self.conv1_1(f1)
        x1 = self.bn1_1(x1)
        x1 = self.relu_1(x1)
        x1 = self.maxpool_1(x1)
        h2_1 = self.res2_1(x1)

        x2 = self.conv1_2(f2)
        x2 = self.bn1_2(x2)
        x2 = self.relu_2(x2)
        x2 = self.maxpool_2(x2)
        h2_2 = self.res2_2(x2)

        x3 = self.conv1_3(f3)
        x3 = self.bn1_3(x3)
        x3 = self.relu_3(x3)
        x3 = self.maxpool_3(x3)
        h2_3 = self.res2_3(x3)

        return h2_1, h2_2, h2_3

    # f1: image1 f2: image2 f3: flow
    def forward(self, f1, f2, f3):
        h2_1, h2_2, h2_3 = self.forward_res2(f1, f2, f3)
        h2_cat1 = torch.cat([h2_1, h2_3], dim=1)
        h2_cat2 = torch.cat([h2_2, h2_3], dim=1)

        # res3 layer: pcam
        h3_1 = self.res3_1(h2_1)
        h3_2 = self.res3_2(h2_2)
        h3_3 = self.res3_3(h2_3)

        Za1, Zb1, Qa1, Qb1 = self.pca_res3(h3_1, h3_3)
        Za2, Zb2, Qa2, Qb2 = self.pca_res3(h3_2, h3_3)
        h3_1 = F.relu(Zb1 + h3_1)
        h3_2 = F.relu(Zb2 + h3_2)
        Qb = self.cca_res3(Qb1, Qb2)    # cross co-attention
        h3_3 = F.relu(Qb + h3_3)
        h3_cat1 = torch.cat([h3_1, h3_3], dim=1)
        h3_cat2 = torch.cat([h3_2, h3_3], dim=1)

        # res4 layer: pcam
        h4_1 = self.res4_1(h3_1)
        h4_2 = self.res4_2(h3_2)
        h4_3 = self.res4_3(h3_3)

        Za1, Zb1, Qa1, Qb1 = self.pca_res4(h4_1, h4_3)
        Za2, Zb2, Qa2, Qb2 = self.pca_res4(h4_2, h4_3)
        h4_1 = F.relu(Zb1 + h4_1)
        h4_2 = F.relu(Zb2 + h4_2)
        Qb = self.cca_res4(Qb1, Qb2)  # cross co-attention
        h4_3 = F.relu(Qb + h4_3)
        h4_cat1 = torch.cat([h4_1, h4_3], dim=1)
        h4_cat2 = torch.cat([h4_2, h4_3], dim=1)

        # res5 layer: pcam
        h5_1 = self.res5_1(h4_1)
        h5_2 = self.res5_2(h4_2)
        h5_3 = self.res5_3(h4_3)

        Za1, Zb1, Qa1, Qb1 = self.pca_res5(h5_1, h5_3)
        Za2, Zb2, Qa2, Qb2 = self.pca_res5(h5_2, h5_3)
        h5_1 = F.relu(Zb1 + h5_1)
        h5_2 = F.relu(Zb2 + h5_2)
        Qb = self.cca_res5(Qb1, Qb2)  # cross co-attention
        h5_3 = F.relu(Qb + h5_3)
        h5_cat1 = torch.cat([h5_1, h5_3], dim=1)
        h5_cat2 = torch.cat([h5_2, h5_3], dim=1)

        # : image1 and flow
        h5_v1 = self.gated_res5(h5_cat1)
        h4_v1 = self.gated_res4(h4_cat1)
        h3_v1 = self.gated_res3(h3_cat1)
        h2_v1 = self.gated_res2(h2_cat1)

        # : image2 and flow
        h5_v2 = self.gated_res5(h5_cat2)
        h4_v2 = self.gated_res4(h4_cat2)
        h3_v2 = self.gated_res3(h3_cat2)
        h2_v2 = self.gated_res2(h2_cat2)

        return h5_v1, h4_v1, h3_v1, h2_v1, h5_v2, h4_v2, h3_v2, h2_v2


# Parallel Co-Attention Module (PCAM)
class ParallelCA(nn.Module):
    def __init__(self, channel):
        super(ParallelCA, self).__init__()
        # project c-dimensional features to multiple lower dimensional spaces
        channel_low = channel // 16

        self.p_f1 = nn.Conv2d(channel, channel_low, kernel_size=1)  # p_f1: project image features
        self.p_f2 = nn.Conv2d(channel, channel_low, kernel_size=1)  # p_f2: project flow features

        self.c_f1 = nn.Conv2d(channel, 1, kernel_size=1)   # c_f1: transform image features to a map by conv 1x1
        self.c_f2 = nn.Conv2d(channel, 1, kernel_size=1)    # c_f2: transform flow features to a map by conv 1x1

        self.relu = nn.ReLU()

    # f1: image, f2: flow
    def forward(self, f1, f2):
        # Stack 1
        f1_1, f2_1 = self.forward_sa(f1, f2)            # soft attention
        f1_hat, f2_hat = self.forward_ca(f1_1, f2_1)    # co-attention

        fp1_hat = F.relu(f1_hat + f1)
        fp2_hat = F.relu(f2_hat + f2)

        # Stack 2
        f1_2, f2_2 = self.forward_sa(fp1_hat, fp2_hat)
        f1_hat, f2_hat = self.forward_ca(f1_2, f2_2)

        fp1_hat = F.relu(f1_hat + fp1_hat)
        fp2_hat = F.relu(f2_hat + fp2_hat)

        # Stack 3
        f1_3, f2_3 = self.forward_sa(fp1_hat, fp2_hat)
        f1_hat, f2_hat = self.forward_ca(f1_3, f2_3)

        fp1_hat = F.relu(f1_hat + fp1_hat)
        fp2_hat = F.relu(f2_hat + fp2_hat)

        # Stack 4
        f1_4, f2_4 = self.forward_sa(fp1_hat, fp2_hat)
        f1_hat, f2_hat = self.forward_ca(f1_4, f2_4)

        fp1_hat = F.relu(f1_hat + fp1_hat)
        fp2_hat = F.relu(f2_hat + fp2_hat)

        # Stack 5
        f1_5, f2_5 = self.forward_sa(fp1_hat, fp2_hat)
        f1_hat, f2_hat = self.forward_ca(f1_5, f2_5)

        return f1_hat, f2_hat, f1_5, f2_5

    # Soft Attention, f1: image and f2: flow
    def forward_sa(self, f1, f2):

        c1 = self.c_f1(f1)  # channel -> 1
        c2 = self.c_f2(f2)  # channel -> 1

        n, c, h, w = c1.shape
        c1 = c1.view(-1, h*w)
        c2 = c2.view(-1, h*w)

        c1 = F.softmax(c1, dim=1)
        c2 = F.softmax(c2, dim=1)

        c1 = c1.view(n, c, h, w)
        c2 = c2.view(n, c, h, w)

        # '*' indicates Hadamard product
        f1_sa = c1 * f1
        f2_sa = c2 * f2

        # f1_sa and f2_sa indicate attention-enhanced features of image and flow, respectively
        return f1_sa, f2_sa

    # Co-Attention, f1: image and f2: flow
    def forward_ca(self, f1, f2):

        f1_cl = self.p_f1(f1)   # f1_cl: dimension from channel to channel_low
        f2_cl = self.p_f2(f2)   # f2_cl: dimension from channel to channel_low

        N, C, H, W = f1_cl.shape
        f1_cl = f1_cl.view(N, C, H * W)
        f2_cl = f2_cl.view(N, C, H * W)
        f2_cl = torch.transpose(f2_cl, 1, 2)

        # Affinity matrix
        A = torch.bmm(f2_cl, f1_cl)

        # A_r: softmax row, A_c: softmax col
        A_c = torch.tanh(A)
        A_r = torch.transpose(A_c, 1, 2)

        N, C, H, W = f1.shape

        f1_v = f1.view(N, C, H * W)
        f2_v = f2.view(N, C, H * W)

        # Co-Attention
        f1_hat = torch.bmm(f1_v, A_r)
        f2_hat = torch.bmm(f2_v, A_c)
        f1_hat = f1_hat.view(N, C, H, W)
        f2_hat = f2_hat.view(N, C, H, W)

        f1_hat = F.normalize(f1_hat)
        f2_hat = F.normalize(f2_hat)

        return f1_hat, f2_hat


# Cross Co-Attention Module (CCAM)
class CrossCoAttention(nn.Module):
    def __init__(self, channel):
        super(CrossCoAttention, self).__init__()

        self.gct = GAC(channel)
        self.faf = GAF(channel)

    def forward(self, f1, f2):
        a1 = self.gct(f1)
        a2 = self.gct(f2)
        aff = self.faf(a1, a2)
        return aff


# Global-local Attention Fusion (GAF)
class GAF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(GAF, self).__init__()
        # r = channels // 16
        inter_channels = int(channels // r)

        # local attention
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # global attention
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # local attention
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # global attention
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


# Global-local Attention Context (GAC)
class GAC(nn.Module):
    def __init__(self, input_channels, eps=1e-5):
        super(GAC, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, input_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, input_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1))
        self.epsilon = eps

    def forward(self, x):
        Nl = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon)
        glo = Nl.pow(0.5) * self.alpha
        Nc = (glo.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        cal = self.gamma / Nc

        v_fea = x * 1. + x * torch.tanh(glo * cal + self.beta)
        return v_fea


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MotionContour(nn.Module):
    def __init__(self, in_channel):
        super(MotionContour, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1,
                               padding=1)
        self.relu = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x


class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        mdim = 256
        self.GC = GC(4096+1, mdim)
        self.convG1 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.convG2 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.RF4 = Refine(2048+1, mdim)
        self.RF3 = Refine(1024+1, mdim)
        self.RF2 = Refine(512+1, mdim)

        self.pred5 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred4 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred3 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.concat = nn.Conv2d(4, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.bdry5 = MotionContour(4096)
        self.bdry4 = MotionContour(2048)
        self.bdry3 = MotionContour(1024)
        self.bdry2 = MotionContour(512)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, h5_1, h4_1, h3_1, h2_1, h5_2, h4_2, h3_2, h2_2):
        # mask1
        c5_1 = self.bdry5(h5_1)
        c4_1 = self.bdry4(h4_1)
        c3_1 = self.bdry3(h3_1)
        c2_1 = self.bdry2(h2_1)

        c2_up_1 = F.interpolate(c2_1, size=(ws, ws), mode='bilinear', align_corners=False)
        c3_up_1 = F.interpolate(c3_1, size=(ws, ws), mode='bilinear', align_corners=False)
        c4_up_1 = F.interpolate(c4_1, size=(ws, ws), mode='bilinear', align_corners=False)
        c5_up_1 = F.interpolate(c5_1, size=(ws, ws), mode='bilinear', align_corners=False)

        concat_1 = torch.cat([c2_up_1, c3_up_1, c4_up_1, c5_up_1], dim=1)
        c_1 = self.concat(concat_1)
        c_1 = torch.sigmoid(c_1)

        h5_1 = torch.cat((h5_1, c5_1), dim=1)
        h4_1 = torch.cat((h4_1, c4_1), dim=1)
        h3_1 = torch.cat((h3_1, c3_1), dim=1)
        h2_1 = torch.cat((h2_1, c2_1), dim=1)

        m_1 = self.forward_mask(h5_1, h4_1, h3_1, h2_1)

        # mask2
        c5_2 = self.bdry5(h5_2)
        c4_2 = self.bdry4(h4_2)
        c3_2 = self.bdry3(h3_2)
        c2_2 = self.bdry2(h2_2)

        c2_up_2 = F.interpolate(c2_2, size=(ws, ws), mode='bilinear', align_corners=False)
        c3_up_2 = F.interpolate(c3_2, size=(ws, ws), mode='bilinear', align_corners=False)
        c4_up_2 = F.interpolate(c4_2, size=(ws, ws), mode='bilinear', align_corners=False)
        c5_up_2 = F.interpolate(c5_2, size=(ws, ws), mode='bilinear', align_corners=False)

        concat_2 = torch.cat([c2_up_2, c3_up_2, c4_up_2, c5_up_2], dim=1)
        c_2 = self.concat(concat_2)
        c_2 = torch.sigmoid(c_2)

        h5_2 = torch.cat((h5_2, c5_2), dim=1)
        h4_2 = torch.cat((h4_2, c4_2), dim=1)
        h3_2 = torch.cat((h3_2, c3_2), dim=1)
        h2_2 = torch.cat((h2_2, c2_2), dim=1)

        m_2 = self.forward_mask(h5_2, h4_2, h3_2, h2_2)

        return m_1, c_1, m_2, c_2

    def forward_mask(self, x, r4, r3, r2):
        x = self.GC(x)
        r = self.convG1(F.relu(x))
        r = self.convG2(F.relu(r))
        m5 = x + r
        m4 = self.RF4(r4, m5)
        m3 = self.RF3(r3, m4)
        m2 = self.RF2(r2, m3)

        p2 = self.pred2(F.relu(m2))
        p2_up = F.interpolate(p2, size=(ws, ws), mode='bilinear', align_corners=False)
        pred = torch.sigmoid(p2_up)

        return pred


class GC(nn.Module):
    def __init__(self, inplanes, planes, kh=7, kw=7):
        super(GC, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, 256, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))
        self.conv_l2 = nn.Conv2d(256, planes, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r1 = nn.Conv2d(inplanes, 256, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r2 = nn.Conv2d(256, planes, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x


class AtrousBlock(nn.Module):
    def __init__(self, inplanes, planes, rate, stride=1):
        super(AtrousBlock, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                              dilation=rate, padding=rate)

    def forward(self, x):
        return self.conv(x)


class PyramidDilationConv(nn.Module):
    def __init__(self, inplanes, planes):
        super(PyramidDilationConv, self).__init__()

        rate = [6, 12, 18]

        self.block0 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.block1 = AtrousBlock(inplanes, planes, rate[0])
        self.block2 = AtrousBlock(inplanes, planes, rate[1])
        self.block3 = AtrousBlock(inplanes, planes, rate[2])
        self.bn = nn.BatchNorm2d(planes*4)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x)
        x2 = self.block1(x)
        x3 = self.block1(x)

        xx = torch.cat([x0, x1, x2, x3], dim=1)
        xx = self.bn(xx)
        return xx


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.convFS2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convFS3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

        out_planes = int(planes / 4)
        self.pdc = PyramidDilationConv(inplanes, out_planes)

    def forward(self, f, pm):
        s = self.pdc(f)
        sr = self.convFS2(F.relu(s))
        sr = self.convFS3(F.relu(sr))
        s = s + sr

        m = s + F.interpolate(pm, size=s.shape[2:4], mode='bilinear', align_corners=False)

        mr = self.convMM1(F.relu(m))
        mr = self.convMM2(F.relu(mr))
        m = m + mr
        return m
