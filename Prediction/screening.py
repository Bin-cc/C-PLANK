import os
import pandas as pd
import torch
from Training.config import get_cfg_defaults
from Prediction.Predictor import PairPredictor
from Training.model import MODEL
from pyteomics import parser
import warnings
warnings.filterwarnings('ignore')


class screening():
    
    def __init__(self, path):

        self.protein_repo = pd.read_csv(os.path.join(path, 'protein_repository.tsv'), sep='\t')
        ligand_repo = pd.read_csv(os.path.join(path, 'ligand_repository.tsv'), sep='\t')
        self.ligand_repo = {x['fragId']:x['SMILES'] for x in ligand_repo.to_dict("records")}

        config = get_cfg_defaults()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = MODEL(device, **config).to(device)
        self.data_path = path
        model.load_state_dict(torch.load(os.path.join(self.data_path, 'best_model.pth')))
        self.predictor = PairPredictor(model, device, self.data_path)


    def ProtProcess(self, prot, prot_type='GeneName', pep_min_len=6, pep_max_len=144):
        if prot_type == 'GeneName':
            try: 
                seq = self.protein_repo[self.protein_repo['Gene Names']==prot]['Sequence'].values[0]
                name = prot
            except: raise ValueError(f"The gene {prot} is not available")
        
        if prot_type == 'UniprotID':
            try: 
                seq = self.protein_repo[self.protein_repo['Entry']==prot]['Sequence'].values[0]
                name = prot
            except: raise ValueError(f"The Uniprot accession {prot} is not available")
        
        if prot_type == 'Sequence':
            seq = prot
            name = 'ProteinA'
        
        pep_lt = parser.cleave(seq, parser.expasy_rules['trypsin'], 0)
        pep_lt = [pep for pep in pep_lt if len(pep)>=pep_min_len and len(pep)<=pep_max_len]
        pep_df = pd.DataFrame(pep_lt,columns=['Peptide'])
        pep_df['start_site'] = pep_df.Peptide.apply(lambda x: seq.find(x)+1)
        pep_df['end_site'] = pep_df['start_site']+pep_df['Peptide'].str.len()-1
        pep_df = pep_df.sort_values(by=['start_site']).reset_index(drop=True)
        
        return seq, name, pep_df
    
    @staticmethod
    def LigProcess(self, lig, lig_type='SMILES'):
        if lig_type == 'SMILES':
            sml = lig
            name = 'LigandA'
        
        if lig_type == 'Repository':
            sml = self.ligand_repo[lig]
            name = lig
        
        return sml, name

    def given_pair_screen(self, prot, ligand, prot_type='GeneName', lig_type='SMILES',
                          mode='all', ligand_name=None):
        seq, prot_name, _ = screening.ProtProcess(self, prot, prot_type)
        sml, lig_name = screening.LigProcess(self, ligand, lig_type)
        if ligand_name is not None: lig_name = ligand_name
        
        if not os.path.exists(os.path.join(self.data_path, 'screening_result')):
            os.mkdir(os.path.join(self.data_path, 'screening_result'))
            
        if mode == 'all':
            out, att_matrix, residue_score, atom_score, diazo_score, alkyne_score, diazo, alkyne, frag_score = self.predictor.single_screen(seq, sml, mode=mode)
            pd.DataFrame(att_matrix).to_csv(os.path.join(self.data_path, f'screening_result/{prot_name}_{lig_name}_att_matrix.csv'), index=False)
            pd.DataFrame(residue_score).to_csv(os.path.join(self.data_path, f'screening_result/{prot_name}_{lig_name}_residue_score.csv'), index=False)
            pd.DataFrame(atom_score).to_csv(os.path.join(self.data_path, f'screening_result/{prot_name}_{lig_name}_atom_score.csv'), index=False)
            
        elif mode == 'single':
            out, diazo_score, alkyne_score, diazo, alkyne, frag_score = self.predictor.single_screen(seq, sml, mode=mode)
        
        return {'Pair score': out[0][0], 'Diazo score': diazo_score, 'Diazo position': diazo,
                'Alkyne score': alkyne_score, 'Alkyne position': alkyne, 'Fragment score': frag_score}

    
    def custom_screen(self, dataset, prot_path=None, batch_size=32, mode='all', project='job'):
        
        if mode == 'all':
            out, residue_score, atom_score, diazo_score, alkyne_score, diazo, alkyne, frag_score = self.predictor.batch_screen(dataset, prot_path=prot_path, project=project,
                                                                                                                          batch_size=batch_size, mode=mode)
            residue_score = pd.DataFrame(residue_score)
            residue_score = pd.concat([out[['Ligand','ProteinID']], residue_score], axis=1)
            residue_score.to_csv(os.path.join(self.data_path, f'screening_result/{project}/residue_score.csv'), index=False)
            atom_score = pd.DataFrame(atom_score)
            atom_score = pd.concat([out[['Ligand','ProteinID']], atom_score], axis=1)
            atom_score.to_csv(os.path.join(self.data_path, f'screening_result/{project}/atom_score.csv'), index=False)
        
        elif mode == 'single':
            out, diazo_score, alkyne_score, diazo, alkyne, frag_score = self.predictor.batch_screen(dataset, prot_path=prot_path, project=project,
                                                                                                    batch_size=batch_size, mode=mode)
        result = {}
        
        for i in out.index:
            prot_name, lig_name = out.loc[i,'ProteinID'], out.loc[i,'Ligand']
            r = {'Pair score': out.loc[i,'Probability'], 'Diazo score': diazo_score[i], 'Diazo position': diazo[i],
                  'Alkyne score': alkyne_score[i], 'Alkyne position': alkyne[i], 'Fragment score': frag_score[i]}
            result.setdefault((lig_name, prot_name), r)
        
        return out, result

