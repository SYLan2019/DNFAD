import math

import torch

import common
import config as c
from model import *
from model_res.extractors import *
from freia_funcs_conv import *
import timm
from efficient import *
import FrEIA.framework as Ff
import FrEIA.modules as Fm

WEIGHT_DIR = './weights'
MODEL_DIR = './models'

class DynamicConv2d(nn.Module):  ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=2,
                 num_groups=8,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim // reduction_ratio, kernel_size=1),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)

        return x.reshape(B, C, H, W)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1):
        super(SpatialAttention, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.idconv = DynamicConv2d(dim=2)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.idconv(x)
        x = self.conv1(x)
        return x * self.sigmoid(x)


class subnet_conv_ln(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_mid = dim_in
        self.attention = SpatialAttention(kernel_size=3)
        self.conv1 = nn.Conv2d(dim_in, dim_mid, 3, 1, 1)
        self.ln = nn.LayerNorm(dim_mid)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(dim_mid, dim_out, 3, 1, 1)

    def forward(self, x):
        x = x + self.attention(x)
        out = self.conv1(x)
        out = self.ln(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.conv2(out)
        return out


def single_parallel_flows(c_feat, c_cond, n_block, clamp_alpha, subnet=subnet_conv_ln):
    flows = Ff.SequenceINN(c_feat, 1, 1)
    print('Build parallel flows: channels:{}, block:{}, cond:{}'.format(c_feat, n_block, c_cond))
    for k in range(n_block):
        flows.append(Fm.AllInOneBlock, cond=0, cond_shape=(c_cond, 1, 1), subnet_constructor=subnet,
                     affine_clamping=clamp_alpha,
                     global_affine_type='SOFTPLUS')
    return flows


def get_nf(input_dim=c.n_feat, channels_hidden=c.channel_hidden):
    nodes = list()
    if c.pos_enc:
        nodes.append(InputNode(c.pos_enc_dim, name='input'))
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks[0]):
        nodes.append(
            Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        if c.pos_enc:
            nodes.append(
                Node([nodes[-1].out0, nodes[0].out0], glow_coupling_layer_cond,
                     {'clamp': c.clamp,
                      'F_class': F_conv,
                      'cond_dim': c.pos_enc_dim,
                      'F_args': {'channels_hidden': channels_hidden,
                                 'kernel_size': c.kernel_sizes}},
                     name=F'conv_{k}'))
        else:
            nodes.append(Node([nodes[-1].out0], glow_coupling_layer_cond,
                              {'clamp': c.clamp,
                               'F_class': F_conv,
                               'F_args': {'channels_hidden': channels_hidden,
                                          'kernel_size': c.kernel_sizes[k]}},
                              name=F'conv_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    nf = ReversibleGraphNet(nodes, n_jac=1)

    return nf


# 32,24,24
def positionalencoding2d(D, H, W):
    """
    taken from https://github.com/gudovskiy/cflow-ad
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    # D = 16
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(np.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P[None].to(c.device)


class ADwithGlow(nn.Module):
    def __init__(self):
        super(ADwithGlow, self).__init__()
        if c.extractor == 'dino':
            self.feature_extractor = timm.create_model(model_name='vit_base_patch8_224_dino', pretrained=True)
            self.input_dims = [768, 768, 768]
            self.feature_extractor.head = nn.Identity()
        elif c.extractor == 'wide_resnet50_2':
            self.feature_extractor, self.input_dims = build_extractor(c)
        elif c.extractor == 'efficient':
            self.feature_extractor = get_pdn_medium()
            state_dict = torch.load(r"./models/EfficientNet/hub/checkpoints/teacher_medium.pth", map_location='cpu')
            self.feature_extractor.load_state_dict(state_dict)
            self.feature_extractor.to(c.device)
            self.input_dims = [384]
        #

        n_feat = {'dino': 768, 'wide_resnet50_2': 1792, 'efficient': 384, 'resnet18': 1024}[c.extractor]
        self.patch_flow = get_nf(input_dim=n_feat)
        self.map_flow = single_parallel_flows(n_feat, c.pos_enc_dim, c.n_coupling_blocks[1], c.clamp)

    def dino_ext(self, x):
        x = self.feature_extractor.patch_embed(x)
        cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
        x = self.feature_extractor.norm_pre(x)
        h_list = []
        for i, blk in enumerate(self.feature_extractor.blocks):
            x = blk(x)
            if i == 7:
                feature = self.feature_extractor.norm(x)
                B, N, C = feature.shape
                feature = feature.transpose(1, 2)
                feature = feature[:, :, 1:]
                H = W = int(math.sqrt(N))
                feature = feature.reshape(B, C, H, W)
                h_list.append(feature)
        return h_list

    def forward(self, x1, x2):
        # map flow
        B, C1, H1, W1 = x1.shape
        B, C2, H2, W2 = x2.shape
        # path flow
        if c.pos_enc:
            pos_enc1 = positionalencoding2d(c.pos_enc_dim, H1, W1)
            cond_1 = pos_enc1.tile(x1.shape[0], 1, 1, 1)
            z1 = self.patch_flow([cond_1, x1])
        else:
            z1 = self.patch_flow(x1)
        jac1 = self.patch_flow.jacobian(run_forward=False)[0]
        # map flow
        if c.pos_enc:
            pos_enc2 = positionalencoding2d(c.pos_enc_dim, H2, W2)
            cond = pos_enc2.tile(x2.shape[0], 1, 1, 1)
            z2, jac2 = self.map_flow(x2, [cond, ])
        else:
            z2, jac2 = self.map_flow(x2)
        return z1, jac1, z2, jac2


def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path, map_location=torch.device('cpu'))
    return model


def save_weights(model, filename):
    model.cpu()
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, filename))
    model.to(c.device)


def load_weights(model, filename):
    path = os.path.join(WEIGHT_DIR, filename)
    model.load_state_dict(torch.load(path))
    model.to(c.device)
    return model
