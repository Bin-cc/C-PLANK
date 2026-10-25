import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from Prepreparing.getEmbedding import ProtEmbedding, LigEmbedding
from training.config import get_cfg_defaults
from training.LoadData import CustomDatasetScreen
from tqdm import tqdm
from multiprocessing import Pool


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
        
        prot_repr = prot_repr.unsqueeze(0).float()
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
            
            att_matrix = torch.mean(att_matrix, 1).squeeze(0).detach().cpu().numpy()
            residue_score = residue_score.detach().cpu().numpy()
            atom_score = atom_score.detach().cpu().numpy()
            
            S = torch.sum(out, dim=-1, keepdim=True)
            p = out / S
            prob = p[:,1]
            S_flat = S.squeeze(-1)
            u = 2.0 / S_flat
            
            if mode == 'single':
                return prob, u
            elif mode == 'all':
                return prob, u, residue_score, atom_score
    

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
        
        out_lt, c_lt = np.array([]), np.array([])
        details = False
        count = 0
        print("Screening dataset:")
        self.model.eval()
        with torch.no_grad():
            for prot_repr, prot_feat, lig_repr, lig_feat, prot_mask, lig_mask in tqdm(dataset):
                prot_repr = prot_repr.to(self.device)
                prot_feat = prot_feat.to(self.device)
                lig_repr = lig_repr.to(self.device)
                lig_feat = lig_feat.to(self.device)
                prot_mask = prot_mask.to(self.device)
                lig_mask = lig_mask.to(self.device)

                
                prot_repr = prot_repr.float()
                out, att_matrix, residue_score, atom_score = self.model(
                                                                    #Protein information
                                                                    prot_repr, prot_feat, prot_mask,
                                                                    #Ligand information
                                                                    lig_repr, lig_feat, lig_mask
                                                                    )
                
                att_matrix = torch.mean(att_matrix, 1)
                for i in range(atom_score.size(0)):
                    if mode == 'all':
                        prot_name, lig_name = out_set.loc[count+i,'ProteinID'], out_set.loc[count+i,'Ligand']
                        att = att_matrix[i, :, :].squeeze(0).detach().cpu().numpy()
                        pd.DataFrame(att).to_csv(os.path.join(self.data_path, f'screening_result/{project}/Attention matrixs/{prot_name}_{lig_name}_att_matrix.csv'), index=False)
                count += batch_size
                
                S = torch.sum(out, dim=-1, keepdim=True)
                p = out / S
                prob = p[:,1]
                S_flat = S.squeeze(-1)
                u = 2.0 / S_flat
                
                if not details:
                    residue_scores = residue_score.clone()
                    atom_scores = atom_score.clone()
                    details = True
                else:
                    residue_scores = torch.cat((residue_scores, residue_score), dim=0)
                    atom_scores = torch.cat((atom_scores, atom_score), dim=0)
                
                out_lt = np.append(out_lt, prob.cpu().tolist())
                c_lt = np.append(c_lt, u.cpu().tolist())
            
            residue_scores = residue_scores.detach().cpu().numpy()
            atom_scores = atom_scores.detach().cpu().numpy()
            
            out_set['Probability'] = out_lt
            out_set['Uncertainty'] = c_lt

            
            if mode == 'single':
                return out_set
            elif mode == 'all':
                return out_set, residue_scores, atom_scores


    def ParseData(self, unique_prot, unique_lig):
        prot_dict, lig_dict = {}, {}
        with Pool(4) as p:
            print("Loading protein information")
            if len(unique_prot) >= 1:
                print("Considering the device memory, protein information will be read one by one  rather than a package")
                prot_dict = None
            else:
                results = p.map(self.pe.prot2embed,tqdm(unique_prot[['ProteinID','Sequence']].values))
                for result in results:
                    prot_dict.update(result)
            print("Loading ligand information")
            results = p.map(self.le.lig2embed,tqdm(unique_lig[['Ligand','SMILES']].values))
            for result in results:
                lig_dict.update(result)
    
        return prot_dict, lig_dict
    
    @staticmethod
    def collate_fn(batch):
        prot_repr, prot_feat, lig_repr, lig_feat, prot_mask, lig_mask = zip(*batch)

        prot_repr = torch.stack(prot_repr)
        prot_feat = torch.stack(prot_feat)
        lig_repr = torch.stack(lig_repr)
        lig_feat = torch.stack(lig_feat)
        prot_mask = torch.stack(prot_mask)
        lig_mask = torch.stack(lig_mask)

        return prot_repr, prot_feat, lig_repr, lig_feat, prot_mask, lig_mask


