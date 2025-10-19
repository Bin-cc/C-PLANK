import os
import pandas as pd
from Training.LoadData import CustomDataset
import torch
from Training.config import get_cfg_defaults
from Training.Trainer import Trainer
from torch.utils.data import DataLoader
from Training.model import MODEL
from sklearn.model_selection import train_test_split
import argparse
import warnings
warnings.filterwarnings('ignore')


def DataSplit(dataset, prot_dict, lig_dict, batch_size=16, random_state=42):
    tv_index, test_index = train_test_split(list(dataset.index), test_size=0.1, random_state=random_state,
                                            shuffle=True, stratify=dataset.label)
    tv_set, test_set = dataset.iloc[tv_index,:], dataset.iloc[test_index,:]
    train_set, val_set = train_test_split(tv_set, test_size=1/9, random_state=random_state,
                                          shuffle=True, stratify=tv_set.label)
    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)
    
    train_set = CustomDataset(train_set, prot_dict, lig_dict)
    val_set = CustomDataset(val_set, prot_dict, lig_dict)
    test_set = CustomDataset(test_set, prot_dict, lig_dict)
    train_set = DataLoader(train_set, shuffle=True, batch_size=batch_size, drop_last=True, 
                           collate_fn=collate_fn)
    val_set = DataLoader(val_set, shuffle=True, batch_size=batch_size, drop_last=False, 
                         collate_fn=collate_fn)
    test_set = DataLoader(test_set, shuffle=True, batch_size=batch_size, drop_last=False, 
                          collate_fn=collate_fn)
    
    return train_set, val_set, test_set


def DataColdSplit(dataset, entity_col, prot_dict, lig_dict, batch_size=16, random_state=42):
    
    entities = dataset[entity_col].unique()
    train_set, tv_index = train_test_split(entities, test_size=0.2, random_state=random_state, shuffle=True)
    val_set, test_set = train_test_split(tv_index, test_size=0.5, random_state=random_state, shuffle=True)
    
    train_set = dataset[dataset[entity_col].isin(train_set)]
    val_set = dataset[dataset[entity_col].isin(val_set)]
    test_set = dataset[dataset[entity_col].isin(test_set)]
    
    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)
    
    train_set = CustomDataset(train_set, prot_dict, lig_dict)
    val_set = CustomDataset(val_set, prot_dict, lig_dict)
    test_set = CustomDataset(test_set, prot_dict, lig_dict)
    train_set = DataLoader(train_set, shuffle=True, batch_size=batch_size, drop_last=True, 
                           collate_fn=collate_fn)
    val_set = DataLoader(val_set, shuffle=True, batch_size=batch_size, drop_last=False, 
                         collate_fn=collate_fn)
    test_set = DataLoader(test_set, shuffle=True, batch_size=batch_size, drop_last=False, 
                          collate_fn=collate_fn)
    
    return train_set, val_set, test_set


def collate_fn(batch):
    prot_repr, prot_feat, lig_repr, lig_feat, prot_mask, lig_mask, diazo_pos, alkyne_pos, label = zip(*batch)

    prot_repr = torch.stack(prot_repr)
    prot_feat = torch.stack(prot_feat)
    lig_repr = torch.stack(lig_repr)
    lig_feat = torch.stack(lig_feat)
    prot_mask = torch.stack(prot_mask)
    lig_mask = torch.stack(lig_mask)
    diazo = torch.stack(diazo_pos)
    alkyne = torch.stack(alkyne_pos)
    
    label = torch.stack(label)
    
    return prot_repr, prot_feat, lig_repr, lig_feat, prot_mask, lig_mask, diazo, alkyne, label


def filter_residue(seq, res=['U','O','X','B','Z','J']):
    return any(r in seq for r in res) == False
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training C-PLANK")
    parser.add_argument('--filepath', required=True, help="path to necessary files", type=str)
    parser.add_argument('--seed', default=42, required=True, type=int, help='Seed for data splitting')
    parser.add_argument('--split', default='random', required=True, type=str, help='Mode of data splitting',
                        choices=['random', 'cold'])
    args = parser.parse_args()
    
    data_path = args.filepath
    seed = args.seed
    split_mode = args.split
    
    dataset = pd.read_csv(os.path.join(data_path, 'split_dataset.tsv'), sep='\t')
    
    #Load protein graphs
    prot_dict = torch.load(os.path.join(data_path, 'prot_dict.pt'))
    #Load ligand graphs
    lig_dict = torch.load(os.path.join(data_path, 'lig_dict.pt'))
    
    if split_mode == 'random':
        train_set, val_set, test_set = DataSplit(dataset, prot_dict, lig_dict,
                                                 batch_size=64, random_state=seed)
    elif split_mode == 'cold':
        train_set, val_set, test_set = DataColdSplit(dataset, 'ProteinID', prot_dict, lig_dict,
                                                     batch_size=64, random_state=seed)
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = get_cfg_defaults()

    model = MODEL(device, **config).to(device)
    
    trainer = Trainer(
                model, epochs=100, optimizer='AdamW', criterion='BCEWithLogitsLoss',
                scheduler='CosineAnnealingLR', device=device, model_path=data_path,
                save_model=True, earlyStop=False, save_path=data_path, lr=config['Optim']['lr'],
                weight_decay=config['Optim']['Wdecay'], score_thre=0.5, evl_name='auc_score'
                      )

    metrics_dict = trainer.model_train(train_set, val_set)
    metrics_dict.to_csv(os.path.join(data_path, 'metrics_output.csv'),index=False)
      
    out_lt, label_lt = trainer.model_test(test_set, model_name='best_model.pth')
    y_pred = (out_lt >= 0.5).astype(int)
    
    test_metrics = trainer.metrics_calculation(y_true=label_lt, y_pred=y_pred, y_prob=out_lt, out_format='frame')     
    test_metrics.to_csv(os.path.join(data_path, 'test_output.csv'),index=False)
