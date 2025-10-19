import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from Training.config import get_cfg_defaults


class CustomDataset(Dataset):
    
    def __init__(self, pairwise, prot_dict, lig_dict):
        self.data = pairwise
        self.prot_dict = prot_dict
        self.lig_dict = lig_dict
        self.config = get_cfg_defaults()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prot = self.data.loc[idx,'ProteinID']
        lig = self.data.loc[idx,'ligID']
        label = self.data.loc[idx,'label']
        
        max_prot = self.config['Utils']['MaxProt']
        max_lig = self.config['Utils']['MaxLig']
        
        #Read in data
        prot_infor = self.prot_dict[prot]
        lig_infor = self.lig_dict[lig]
            
        prot_nodes = prot_infor['token_repr'].size(0)
        lig_nodes = lig_infor['atom_repr'].size(0)
        if prot_nodes > max_prot:
            prot_repr = prot_infor['token_repr'][:max_prot, :]
            prot_feat = prot_infor['phychem_feat'][:max_prot, :]
        else:
            prot_repr = F.pad(prot_infor['token_repr'], (0, 0, 0, max_prot-prot_nodes), value=0)
            prot_feat = F.pad(prot_infor['phychem_feat'], (0, 0, 0, max_prot-prot_nodes), value=0)
        
        if lig_nodes > max_lig:
            lig_repr = lig_infor['atom_repr'][:max_lig, :]
            lig_feat = lig_infor['atom_feats'][:max_lig, :]
        else:
            lig_repr = F.pad(lig_infor['atom_repr'], (0, max_lig-lig_nodes), value=0)
            lig_feat = F.pad(lig_infor['atom_feats'], (0, 0, 0, max_lig-lig_nodes), value=0)
        
        prot_mask = (prot_repr.abs().sum(dim=1) != 0).long()
        lig_mask = (lig_feat.abs().sum(dim=1) != 0).long()
        
        diazo_pos = torch.tensor(lig_infor['diazo_pos'][0])
        alkyne_pos = torch.tensor(lig_infor['alkyne_pos'][0])
        ## Y output
        label = torch.tensor(label).long()
        
        return prot_repr, prot_feat, lig_repr, lig_feat, prot_mask, lig_mask, diazo_pos, alkyne_pos, label


class CustomDatasetScreen(Dataset):
    
    def __init__(self, pairwise, prot_dict, lig_dict, prot_path=None):
        self.data = pairwise
        if prot_dict is None:
            self.prot_path = prot_path
        self.prot_dict = prot_dict
        self.lig_dict = lig_dict
        self.config = get_cfg_defaults()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prot = self.data.loc[idx,'ProteinID']
        lig = self.data.loc[idx,'Ligand']
        
        max_prot = self.config['Utils']['MaxProt']
        max_lig = self.config['Utils']['MaxLig']
        
        #Read in data
        if self.prot_dict is None:
            prot_infor = torch.load(os.path.join(self.prot_path, f'{prot}.pt'))
        else: prot_infor = self.prot_dict[prot]
        lig_infor = self.lig_dict[lig]
            
        prot_nodes = prot_infor['token_repr'].size(0)
        lig_nodes = lig_infor['atom_repr'].size(0)
        if prot_nodes > max_prot:
            prot_repr = prot_infor['token_repr'][:max_prot, :]
            prot_feat = prot_infor['phychem_feat'][:max_prot, :]
        else:
            prot_repr = F.pad(prot_infor['token_repr'], (0, 0, 0, max_prot-prot_nodes), value=0)
            prot_feat = F.pad(prot_infor['phychem_feat'], (0, 0, 0, max_prot-prot_nodes), value=0)
        
        if lig_nodes > max_lig:
            lig_repr = lig_infor['atom_repr'][:max_lig, :]
            lig_feat = lig_infor['atom_feats'][:max_lig, :]
        else:
            lig_repr = F.pad(lig_infor['atom_repr'], (0, max_lig-lig_nodes), value=0)
            lig_feat = F.pad(lig_infor['atom_feats'], (0, 0, 0, max_lig-lig_nodes), value=0)
        
        prot_mask = (prot_repr.abs().sum(dim=1) != 0).long()
        lig_mask = (lig_feat.abs().sum(dim=1) != 0).long()
        
        diazo_pos = torch.tensor(lig_infor['diazo_pos'][0])
        alkyne_pos = torch.tensor(lig_infor['alkyne_pos'][0])
        
        return prot_repr, prot_feat, lig_repr, lig_feat, prot_mask, lig_mask, diazo_pos, alkyne_pos



