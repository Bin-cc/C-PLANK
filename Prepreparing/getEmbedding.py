import os
import esm
import math
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
RDLogger.DisableLog('rdApp.warning')

class ProtEmbedding():

    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        source_path=r"D:\All_for_paper\1. PhD Work Program\3. Research project\2. Ligand Discovery\public data source\Structural information"
        self.feat = pd.read_excel(os.path.join(source_path, 'physicochemical features.xlsx'))
        self.feat_descriptor = {}
        for i in self.feat.index:
            self.feat_descriptor.setdefault(self.feat.loc[i,'Residue'],tuple((self.feat.iloc[i,1:].values.astype(np.float64))))
        
        self.one2three = {
            'V': 'VAL', 'I': 'ILE', 'L': 'LEU', 'E': 'GLU', 'Q': 'GLN',
            'D': 'ASP', 'N': 'ASN', 'H': 'HIS', 'W': 'TRP', 'F': 'PHE',
            'Y': 'TYR', 'R': 'ARG', 'K': 'LYS', 'S': 'SER', 'T': 'THR',
            'M': 'MET', 'A': 'ALA', 'G': 'GLY', 'P': 'PRO', 'C': 'CYS'
            }
    
    
    @staticmethod
    def esm_feat(self, seq, layer=33, dim=1280):
        prot_id = 'prot_id'
        if len(seq) <= 1000:
            data = [(prot_id, seq)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(next(self.model.parameters()).device, non_blocking=True)
    
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[i for i in range(6,layer+1)], return_contacts=True)
            
            logits = results["logits"][0].detach().cpu().numpy()[1: len(seq) + 1]
            contact_map = results["contacts"][0].detach().cpu().numpy()
            token_repr = torch.cat([results['representations'][i] for i in range(6,layer+1)])
            assert token_repr.size(0) == len(range(6,layer+1))
            token_repr = token_repr.mean(dim=0)
            token_repr = token_repr.detach().cpu().numpy()
            token_repr = token_repr[1: len(seq) + 1]
        else:
            contact_map = np.zeros((len(seq), len(seq)))  # global contact map prediction
            token_repr = np.zeros((len(seq), dim))
            logits = np.zeros((len(seq),layer))
            
            interval = 500
            i = math.ceil(len(seq) / interval)
            # ======================
            # =                    =
            # =                    =
            # =          ======================
            # =          =*********=          =
            # =          =*********=          =
            # ======================          =
            #            =                    =
            #            =                    =
            #            ======================
            # where * is the overlapping area
            # subsection seq contact map prediction
            for s in range(i):
                start = s*interval  # sub seq predict start
                end = min((s+2)*interval, len(seq))  # sub seq predict end
                
                # prediction
                temp_seq = seq[start:end]
                temp_data = [(prot_id, temp_seq)]
                batch_labels, batch_strs, batch_tokens = self.batch_converter(temp_data)
                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[i for i in range(6,layer+1)], return_contacts=True)
                
                # insert into the global contact map
                row, col = np.where(contact_map[start:end, start:end] != 0)
                row = row + start
                col = col + start
                contact_map[start:end,start:end] = contact_map[start:end,start:end]+results["contacts"][0].detach().cpu().numpy()
                contact_map[row, col] = contact_map[row, col]/2.0
                logits[start:end] += results['logits'][0].detach().cpu().numpy()[1: len(temp_seq)+1]
                logits[row] = logits[row]/2.0
                subtoken_repr = torch.cat([results['representations'][i] for i in range(6, layer+1)])
                assert subtoken_repr.size(0) == len(range(6,layer+1))
                subtoken_repr = subtoken_repr.mean(dim=0)
                subtoken_repr = subtoken_repr.detach().cpu().numpy()[1: len(temp_seq) + 1]
                trow = np.where(token_repr[start:end].sum(axis=-1) != 0)[0]
                trow = trow + start
                token_repr[start:end] = token_repr[start:end] + subtoken_repr
                token_repr[trow] = token_repr[trow]/2.0
                    
                if end == len(seq):
                    break

        return torch.from_numpy(token_repr), torch.from_numpy(contact_map), torch.from_numpy(logits)
        
    
    def prot2embed(self, values):
        prot, prot_seq = values[0], values[1]
        token_repr, _, _ = ProtEmbedding.esm_feat(self, prot_seq)
    
        phychem = list(map(lambda x:self.feat_descriptor[self.one2three[x]],list(prot_seq)))
        
        prot_embed = {
            'phychem_feat':torch.tensor(phychem, dtype=torch.float32).clone(),
            'token_repr':token_repr.clone(),
            'prot_len':len(prot_seq)
            }
        
        return {prot : prot_embed}


class LigEmbedding():
    
    def __init__(self):
        
        self.atom_num = {'C':1, 'N':2, 'O':3, 'F':4, 'P':5,
                         'S':6, 'Cl':7, 'Br':8, 'I':9}
        
        self.diazo = Chem.MolFromSmiles('C-N=N')
        self.alkyne = Chem.MolFromSmiles('C#C-C-C')
    
    
    @staticmethod
    def standardize_smi(sml):
        mol = Chem.MolFromSmiles(sml)
        try: clean_mol = rdMolStandardize.Cleanup(mol) # 除去氢、金属原子，标准化分子
        except: clean_mol = mol
        stan_smiles = Chem.MolToSmiles(clean_mol, canonical=True)
        
        return stan_smiles
    
    
    def lig2embed(self, value):
        ligId = value[0]
        smiles = value[1]
        stan_sml = LigEmbedding.standardize_smi(smiles)
        mol = Chem.MolFromSmiles(stan_sml)
        if mol is None:
            raise ValueError("Invalid SMILES code: %s" % (smiles))
        
        atom_feats = LigEmbedding.feature_extract(mol)

        atom_repr = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in self.atom_num.keys():
                atom_repr.append(self.atom_num[atom.GetSymbol()])
            else: atom_repr.append(0)

        diazo_pos = mol.GetSubstructMatches(self.diazo)
        alkyne_pos = mol.GetSubstructMatches(self.alkyne)
        
        lig_embed = {
            'atom_feats': torch.tensor(atom_feats, dtype=torch.float32).clone(),
            'atom_repr': torch.tensor(atom_repr, dtype=torch.long).clone(),
            'lig_len': mol.GetNumAtoms(),
            'diazo_pos': diazo_pos,
            'alkyne_pos': alkyne_pos
            }
        
        return {ligId : lig_embed}
    
        
    @staticmethod
    def feature_extract(mol):
        (logP, mr) = zip(*(rdDesc._CalcCrippenContribs(mol)))
        logP, mr = list(logP), list(mr)
        tpsa = list(rdDesc._CalcTPSAContribs(mol))
        asa, _ = rdDesc._CalcLabuteASAContribs(mol)
        asa = list(asa)
        feat_values = np.array([logP,mr,tpsa,asa]).T
        charges = LigEmbedding.charge_cal(mol)
        feat_values = np.hstack((feat_values,np.array(charges).reshape(mol.GetNumAtoms(),1)))
        h_num = [mol.GetAtomWithIdx(i).GetTotalNumHs() for i in range(mol.GetNumAtoms())]
        h_num  = list(map(lambda x:LigEmbedding.one_hot_k(x, list(range(8))), h_num))
        feat_values = np.hstack((feat_values,h_num))
        d_num = [mol.GetAtomWithIdx(i).GetDegree() for i in range(mol.GetNumAtoms())]
        d_num  = list(map(lambda x:LigEmbedding.one_hot_k(x, list(range(8))), d_num))
        feat_values = np.hstack((feat_values,d_num))
        i_num = [mol.GetAtomWithIdx(i).GetImplicitValence() for i in range(mol.GetNumAtoms())]
        i_num  = list(map(lambda x:LigEmbedding.one_hot_k(x, list(range(8))), i_num))
        feat_values = np.hstack((feat_values,i_num))
        hybrid = [mol.GetAtomWithIdx(i).GetHybridization() for i in range(mol.GetNumAtoms())]
        hybrid_type = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other']
        hybrid  = list(map(lambda x:LigEmbedding.one_hot_k(x, hybrid_type), hybrid))
        feat_values = np.hstack((feat_values,hybrid))
        stereo = Chem.FindMolChiralCenters(mol,includeUnassigned=True)
        chiral_centers = [0] * mol.GetNumAtoms()
        for i in stereo: chiral_centers[i[0]] = 1
        feat_values = np.hstack((feat_values,np.array(chiral_centers).reshape(mol.GetNumAtoms(),1)))
        key_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'other']
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atoms  = list(map(lambda x:LigEmbedding.one_hot_k(x, key_atoms), atoms))
        feat_values = np.hstack((feat_values,atoms))
        
        return feat_values
    
    @staticmethod
    def charge_cal(mol):
        mol_with_hs = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol_with_hs)
        charges = rdDesc.CalcEEMcharges(mol_with_hs)
        
        charges_without_hs = []
        for i,atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() != 'H':
                charges_without_hs.append(charges[i])
        return charges_without_hs
    
    @staticmethod
    def one_hot_k(x, num_list):
        if x not in num_list: x == 'other'
        return [1 if x == num else 0 for num in num_list]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embedding generation for C-PLANK")
    parser.add_argument('--outpath', required=True, help="path to output files", type=str)
    parser.add_argument('--datapath', required=True, type=str, help='path to dataset')
    args = parser.parse_args()
    
    dataset = pd.read_csv(args.datapath, sep='\t')
    
    prot_infor = dataset.drop_duplicates(subset='ProteinID')[['ProteinID','sequence']].values
    lig_infor = dataset.drop_duplicates(subset='ligID')[['ligID','SMILES']].values
    pe = ProtEmbedding()
    le = LigEmbedding()
    
    with Pool(16) as p:
        prot_result = p.map(pe.prot2embed,tqdm(prot_infor))
        lig_result = p.map(le.lig2embed,tqdm(lig_infor))

    prot_dict = {result for result in prot_result}
    lig_dict = {result for result in lig_result}
    
    torch.save(prot_dict, os.path.join(args.outpath, 'prot_dict.pt'))
    torch.save(lig_dict, os.path.join(args.outpath, 'lig_dict.pt'))
