import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from Prepreparing.getEmbedding import ProtEmbedding, LigEmbedding
from Training.config import get_cfg_defaults
from Training.LoadData import CustomDatasetScreen
from tqdm import tqdm


class PairPredictor():
    
    def __init__(self, model, device, data_path):
        
        self.model = model
        self.device = device
        self.pe = ProtEmbedding()
        self.le = LigEmbedding()
        self.config = get_cfg_defaults()
        self.scale = nn.Sigmoid()
        self.data_path = data_path
        
    
    def single_screen(self, prot, ligand, mode='single', prot_name=None, lig_name=None):
        if prot_name is not None: prot_name = prot_name
        else: prot_name = 'prot'
        
        if lig_name is not None: lig_name = lig_name
        else: lig_name = 'ligand'
        
        prot_infor = self.pe.prot2embed((prot_name, prot))
        lig_infor = self.le.lig2embed((lig_name, ligand))
        max_prot = self.config['Utils']['MaxProt']
        max_lig = self.config['Utils']['MaxLig']
        
        prot_repr = prot_infor[prot_name]['token_repr']
        prot_feat = prot_infor[prot_name]['phychem_feat']
        prot_len = prot_infor[prot_name]['prot_len']
        lig_repr = lig_infor[lig_name]['atom_repr']
        lig_feat = lig_infor[lig_name]['atom_feats']
        lig_len = lig_infor[lig_name]['lig_len']
        
        
        if prot_len > max_prot:
            prot_repr = prot_repr[:max_prot, :]
            prot_feat = prot_feat[:max_prot, :]
        else:
            prot_repr = F.pad(prot_repr, (0, 0, 0, max_prot-prot_len), value=0)
            prot_feat = F.pad(prot_feat, (0, 0, 0, max_prot-prot_len), value=0)
        
        if lig_len > max_lig:
            lig_repr = lig_repr[:max_lig, :]
            lig_feat = lig_feat[:max_lig, :]
        else:
            lig_repr = F.pad(lig_repr, (0, max_lig-lig_len), value=0)
            lig_feat = F.pad(lig_feat, (0, 0, 0, max_lig-lig_len), value=0)
        
        prot_mask = (prot_repr.abs().sum(dim=1) != 0).long()
        lig_mask = (lig_feat.abs().sum(dim=1) != 0).long()
        
        prot_repr = prot_repr.to(self.device)
        prot_feat = prot_feat.to(self.device)
        lig_repr = lig_repr.to(self.device)
        lig_feat = lig_feat.to(self.device)
        prot_mask = prot_mask.to(self.device)
        lig_mask = lig_mask.to(self.device)
        diazo = lig_infor[lig_name]['diazo_pos']
        alkyne = lig_infor[lig_name]['alkyne_pos']
        
        prot_repr = prot_repr.unsqueeze(0)
        prot_repr = prot_repr.float()
        prot_feat = prot_feat.unsqueeze(0)
        prot_mask = prot_mask.unsqueeze(0)
        lig_repr = lig_repr.unsqueeze(0)
        lig_feat = lig_feat.unsqueeze(0)
        lig_mask = lig_mask.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            out, att_matrix, residue_score, atom_score = self.model(
                                                                #Protein information
                                                                prot_repr, prot_feat, prot_mask,
                                                                #Ligand information
                                                                lig_repr, lig_feat, lig_mask
                                                                )
            
            diazo_score = atom_score[:, list(diazo[0])].mean()
            alkyne_score = atom_score[:, list(alkyne[0])].mean()
            
            frag_mask = torch.ones(atom_score.size(1), dtype=torch.bool)
            frag_mask[list(diazo[0])+list(alkyne[0])] = False
            frag_score = atom_score[:, frag_mask].sum()/(lig_len-len(diazo[0])-len(alkyne[0]))
            
            score = self.scale(out).detach().cpu().numpy()
            diazo_score = diazo_score.detach().cpu().numpy()
            alkyne_score = alkyne_score.detach().cpu().numpy()
            frag_score = frag_score.detach().cpu().numpy()
            att_matrix = torch.mean(att_matrix, 1).squeeze(0).detach().cpu().numpy()
            residue_score = residue_score.detach().cpu().numpy()
            atom_score = atom_score.detach().cpu().numpy()
            
            
            if mode == 'single':
                return score, diazo_score, alkyne_score, list(diazo[0]), list(alkyne[0]), frag_score
            elif mode == 'all':
                return score, att_matrix, residue_score, atom_score, diazo_score, alkyne_score, list(diazo[0]), list(alkyne[0]), frag_score
    

    def batch_screen(self, dataset, prot_path=None, batch_size=32, mode='single', project='job'):
        unique_prot = dataset.drop_duplicates(subset=['ProteinID'])
        unique_lig = dataset.drop_duplicates(subset=['Ligand'])
        prot_dict, lig_dict = self.ParseData(unique_prot, unique_lig)
        out_set = dataset[['Ligand', 'ProteinID']].copy()
        
        dataset = CustomDatasetScreen(dataset, prot_dict, lig_dict, prot_path=prot_path)
        dataset = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=False, 
                             collate_fn=PairPredictor.collate_fn)
        
        if not os.path.exists(os.path.join(self.data_path, 'screening_result')):
            os.mkdir(os.path.join(self.data_path, 'screening_result'))
        os.mkdir(os.path.join(self.data_path, f'screening_result/{project}'))
        if mode == 'all':
            os.mkdir(os.path.join(self.data_path, f'screening_result/{project}/Attention matrixs'))
        
        out_lt, diazo_score, alkyne_score, frag_score = np.array([]), np.array([]), np.array([]), np.array([])
        details = False
        count = 0
        print("Screening dataset:")
        self.model.eval()
        with torch.no_grad():
            for prot_repr, prot_feat, lig_repr, lig_feat, prot_mask, lig_mask, diazo, alkyne in tqdm(dataset):
                prot_repr = prot_repr.to(self.device)
                prot_feat = prot_feat.to(self.device)
                lig_repr = lig_repr.to(self.device)
                lig_feat = lig_feat.to(self.device)
                prot_mask = prot_mask.to(self.device)
                lig_mask = lig_mask.to(self.device)
                diazo = diazo.to(self.device)
                alkyne = alkyne.to(self.device)
                
                prot_repr = prot_repr.float()
                out, att_matrix, residue_score, atom_score = self.model(
                                                                    #Protein information
                                                                    prot_repr, prot_feat, prot_mask,
                                                                    #Ligand information
                                                                    lig_repr, lig_feat, lig_mask
                                                                    )
                
                att_matrix = torch.mean(att_matrix, 1)
                for i in range(atom_score.size(0)):
                    d_score = atom_score[i, diazo[i]].mean()
                    a_score = atom_score[i, alkyne[i]].mean()
                    frag_mask = torch.ones(atom_score.size(1), dtype=torch.bool)
                    frag_mask[torch.cat((diazo[i], alkyne[i]), dim=0)] = False
                    f_score = atom_score[i, frag_mask].sum()/(lig_mask[i].sum()-len(diazo[i])-len(alkyne[i]))
                    diazo_score = np.append(diazo_score, d_score.detach().cpu().numpy())
                    alkyne_score = np.append(alkyne_score, a_score.detach().cpu().numpy())
                    frag_score = np.append(frag_score, f_score.detach().cpu().numpy())
                    
                    if mode == 'all':
                        prot_name, lig_name = out_set.loc[count+i,'ProteinID'], out_set.loc[count+i,'Ligand']
                        att = att_matrix[i, :, :].squeeze(0).detach().cpu().numpy()
                        pd.DataFrame(att).to_csv(os.path.join(self.data_path, f'screening_result/{project}/Attention matrixs/{prot_name}_{lig_name}_att_matrix.csv'), index=False)
                count += batch_size
                
                score = self.scale(out).detach().cpu().numpy()
                
                if not details:
                    residue_scores = residue_score.clone()
                    atom_scores = atom_score.clone()
                    diazo_pos = diazo.clone()
                    alkyne_pos = alkyne.clone()
                    details = True
                else:
                    residue_scores = torch.cat((residue_scores, residue_score), dim=0)
                    atom_scores = torch.cat((atom_scores, atom_score), dim=0)
                    diazo_pos = torch.cat((diazo_pos, diazo), dim=0)
                    alkyne_pos = torch.cat((alkyne_pos, alkyne), dim=0)
                
                out_lt = np.append(out_lt, score)
            
            residue_scores = residue_scores.detach().cpu().numpy()
            atom_scores = atom_scores.detach().cpu().numpy()
            diazo_pos = diazo_pos.detach().cpu().numpy()
            alkyne_pos = alkyne_pos.detach().cpu().numpy()
            
            out_set['Probability'] = out_lt
            
            if mode == 'single':
                return out_set, diazo_score, alkyne_score, diazo_pos, alkyne_pos, frag_score
            elif mode == 'all':
                return out_set, residue_scores, atom_scores, diazo_score, alkyne_score, diazo_pos, alkyne_pos, frag_score

    def ParseData(self, unique_prot, unique_lig):
        prot_dict, lig_dict = {}, {}
        
        print("Loading protein information")
        for protId, seq in tqdm(unique_prot[['ProteinID','Sequence']].values):
            result = self.pe.prot2embed((protId, seq))
            prot_dict.update(result)
        
        print("Loading ligand information")
        for ligId, sml in tqdm(unique_prot[['Ligand','SMILES']].values):
            result = self.le.lig2embed((ligId, sml))
            lig_dict.update(result)
        
        return prot_dict, lig_dict
    
    @staticmethod
    def collate_fn(batch):
        prot_repr, prot_feat, lig_repr, lig_feat, prot_mask, lig_mask, diazo_pos, alkyne_pos = zip(*batch)

        prot_repr = torch.stack(prot_repr)
        prot_feat = torch.stack(prot_feat)
        lig_repr = torch.stack(lig_repr)
        lig_feat = torch.stack(lig_feat)
        prot_mask = torch.stack(prot_mask)
        lig_mask = torch.stack(lig_mask)
        diazo = torch.stack(diazo_pos)
        alkyne = torch.stack(alkyne_pos)
        
        return prot_repr, prot_feat, lig_repr, lig_feat, prot_mask, lig_mask, diazo, alkyne


