import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F


class CPLANK(nn.Module):

    def __init__(self, device, **config):
        super(CPLANK, self).__init__()
        
        dropout_p = config['Utils']['Dropout']
        self.device = device
        self.dropout = nn.Dropout(p=dropout_p)
        self.max_prot = config['Utils']['MaxProt']
        self.max_lig = config['Utils']['MaxLig']
        node_dim = config['Utils']['NodeDim']
        
        #Protein feature extractor
        prot_repr_dim = config['Protein']['ReprDim']
        prot_feat_dim = config['Protein']['FeatDim']
        prot_bias = config['Protein']['Bias']
        prot_masked = config['Protein']['Mask']
        
        self.prot_repr_extractor = MLPBlock(dims=[prot_repr_dim, node_dim],
                                            bias=prot_bias, masked=prot_masked)
        
        self.prot_feat_extractor = MLPBlock(dims=[prot_feat_dim, node_dim],
                                            bias=prot_bias, masked=prot_masked)
        
        #Ligand feature extractor
        lig_repr_dim = config['Ligand']['ReprDim']
        lig_feat_dim = config['Ligand']['FeatDim']
        lig_bias = config['Ligand']['Bias']
        lig_masked = config['Ligand']['Mask']
        
        self.lig_embed_node = nn.Embedding(num_embeddings=15, 
                                           embedding_dim=lig_repr_dim, 
                                           padding_idx=0)
        
        self.lig_repr_extractor = MLPBlock(dims=[lig_repr_dim, node_dim],
                                           bias=lig_bias, masked=lig_masked)
        
        self.lig_feat_extractor = MLPBlock(dims=[lig_feat_dim, node_dim],
                                           bias=lig_bias, masked=lig_masked)
       
        #CNN encoder
        channels = config['CNN']['Channels']
        prot_kernels = config['CNN']['ProtKernels']
        lig_kernels = config['CNN']['LigKernels']
        
        self.prot_cnn = CNNBlock([node_dim]+channels, prot_kernels)
        self.lig_cnn = CNNBlock([node_dim]+channels, lig_kernels)
        self.p_net = nn.MaxPool1d(self.max_prot)
        self.l_net = nn.MaxPool1d(self.max_lig)
        
        #BAN Attention
        h_dim = channels[-1]*2
        h_out = config['BAN']['Hout']
        
        self.prot_att_layer = nn.Linear(channels[-1], channels[-1])
        self.lig_att_layer = nn.Linear(channels[-1], channels[-1])
        self.ban = BANLayer(v_dim=channels[-1], q_dim=channels[-1],
                            h_dim=h_dim, h_out=h_out, dropout=dropout_p)
        
        #MLP decoder
        self.prot_att_dec = PosLinear(in_dim=h_out, out_dim=1)
        self.lig_att_dec = PosLinear(in_dim=h_out, out_dim=1)
        dec_dim = config['Decoder']['Dims']
        
        out_channel = h_dim
        self.global_enc = nn.Linear(out_channel, out_channel)
        self.predict = MLPBlock(dims=[out_channel*2]+dec_dim,
                                dropout_p=dropout_p, act='LeakyReLU', binary=True)
        
    def forward(self, 
                #Protein features
                prot_repr, prot_feats, prot_mask,
                #Ligand features
                lig_repr, lig_feats, lig_mask,
                #Batch
                prot_batch=None, lig_batch=None,
                ):
            
        #Protein feature processing
        prot_repr = self.prot_repr_extractor(prot_repr, prot_mask)
        prot_feats = self.prot_feat_extractor(prot_feats, prot_mask)
        prot_embed = prot_repr+prot_feats
        
        prot_embed = prot_embed.permute(0, 2, 1)
        prot_conv = self.prot_cnn(prot_embed)
        prot_att = self.prot_att_layer(prot_conv.permute(0, 2, 1))
        prot_pool = self.p_net(prot_conv).squeeze(2)
        
        #Ligand feature processing
        lig_repr = self.lig_embed_node(lig_repr.squeeze())
        lig_repr = self.lig_repr_extractor(lig_repr, lig_mask)
        lig_feats = self.lig_feat_extractor(lig_feats, lig_mask)
        lig_embed = lig_repr+lig_feats
        
        lig_embed = lig_embed.permute(0, 2, 1)
        lig_conv = self.lig_cnn(lig_embed)
        lig_att = self.lig_att_layer(lig_conv.permute(0, 2, 1))
        lig_pool = self.l_net(lig_conv).squeeze(2)
        
        #Interaction
        logits, att_matrix = self.ban(lig_att, prot_att)
        atom_score = torch.mean(att_matrix, 3)
        residue_score = torch.mean(att_matrix, 2)
        
        residue_score = residue_score.transpose(1, 2)
        residue_score = self.prot_att_dec(F.sigmoid(residue_score))
        residue_score = F.softmax(residue_score.squeeze(-1).masked_fill(~(prot_mask.bool()), -1e10), dim=1)
        atom_score = atom_score.transpose(1, 2)
        atom_score = self.lig_att_dec(F.sigmoid(atom_score))
        atom_score = F.softmax(atom_score.squeeze(-1).masked_fill(~(lig_mask.bool()), -1e10), dim=1)
        
        pair = torch.cat([lig_pool, prot_pool], dim=1)
        global_att = F.leaky_relu(self.global_enc(pair))
        pair = torch.cat([global_att, logits], dim=1)
        out = self.predict(pair)
        
        return out, att_matrix, residue_score, atom_score


class MLPBlock(nn.Module):
    
    def __init__(self, dims, dropout_p=0, act='ReLU', bias=True, masked=False,
                 l_norm=False, w_norm=False, binary=False):
        super(MLPBlock, self).__init__()
        
        self.mlp = nn.ModuleList()
        for i in range(len(dims)-1):
            self.mlp.append(nn.Linear(dims[i], dims[i+1], bias=bias))
        
        self.act = getattr(nn, act)()
        self.out = nn.Linear(dims[-1], 1)
        self.Nlayers = len(self.mlp)
        self.dropout = nn.Dropout(p=dropout_p)
        self.norm_layer = nn.LayerNorm(dims[-1])
        self.w_norm = w_norm
        self.l_norm = l_norm
        self.bias = bias
        self.masked = masked
        self.dropout_p = dropout_p
        self.binary = binary
        
    def forward(self, X, mask=None):
        for idx in range(self.Nlayers):
            if self.dropout_p>0:
                X = self.dropout(X)
            if self.w_norm:
                X = weight_norm(self.mlp[idx](X))
            else:
                X = self.mlp[idx](X)
            if self.bias and self.masked:
                X = X*mask.unsqueeze(-1)
            
            if idx == self.Nlayers-1:
                if self.l_norm:
                    X = self.norm_layer(X)
                continue
            X = self.act(X)

        if self.binary:
            return self.out(X)
        else:    
            return X

class CNNBlock(nn.Module):
    
    def __init__(self, channels, kernels):
        super(CNNBlock, self).__init__()
        
        convs = []
        for i in range(len(channels)-1):
            convs.append(nn.Conv1d(channels[i], channels[i+1], 
                                   kernel_size=kernels[i], padding=kernels[i]//2))
            convs.append(nn.BatchNorm1d(channels[i+1]))
            convs.append(nn.ReLU())
        self.convs = nn.Sequential(*convs)

        if channels[0] != channels[-1]:
            self.res_layer = nn.Conv1d(channels[0], channels[-1], kernel_size=1)
        else:
            self.res_layer = None
    
    def forward(self, X):
        res = X.clone()
        X = self.convs(X)
        res = res if self.res_layer is None else self.res_layer(res)
        X = X+res
        
        return X
    
class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i

        logits = self.bn(logits)
        return logits, att_maps


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class PosLinear(nn.Module):
    """
    Modified from https://github.com/huankoh/PSICHIC/blob/main/models/layers.py
    """
    
    def __init__(self, in_dim, out_dim, eps=1e-5, bias=False):
        super(PosLinear, self).__init__()
        
        weight = nn.init.uniform_(torch.empty((out_dim, in_dim)), a=eps, b=1)
        self.weight = nn.Parameter(weight.log())
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, input):
        return F.relu(F.linear(input, F.softmax(self.weight.exp(), dim=-1), self.bias))
  
