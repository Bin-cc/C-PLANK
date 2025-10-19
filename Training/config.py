from yacs.config import CfgNode as CN

cfg = CN()
cfg.Utils = CN()
cfg.Utils.Dropout = 0.1
cfg.Utils.MaxProt = 1200
cfg.Utils.MaxLig = 60
cfg.Utils.NodeDim = 64

#Protein feature extractor
cfg.Protein = CN()
cfg.Protein.ReprDim = 1280
cfg.Protein.FeatDim = 26
cfg.Protein.Mask = True
cfg.Protein.Bias = True

#Ligand feature extractor
cfg.Ligand = CN()
cfg.Ligand.ReprDim = 64
cfg.Ligand.FeatDim = 43
cfg.Ligand.Mask = True
cfg.Ligand.Bias = True

#CNN encoder
cfg.CNN = CN()
cfg.CNN.Channels = [32, 64, 128]
cfg.CNN.ProtKernels = [3, 7, 11]
cfg.CNN.LigKernels = [3, 7, 11]

#BAN decoder
cfg.BAN = CN()
cfg.BAN.Hout = 4

#MLP decoder
cfg.Decoder = CN()
cfg.Decoder.Dims = [512, 512, 128]

#Optimizer
cfg.Optim = CN()
cfg.Optim.lr = 1e-4
cfg.Optim.Wdecay = 1e-4

def get_cfg_defaults():
    return cfg.clone()