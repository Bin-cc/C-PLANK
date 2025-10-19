import torch
import torch.nn as nn
from prefetch_generator import BackgroundGenerator
from Training.EarlyStopping import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score,accuracy_score,matthews_corrcoef,average_precision_score,auc,precision_recall_curve
from tqdm import tqdm
import os
import copy


class Trainer():
    
    def __init__(self, model, epochs, optimizer, criterion, scheduler, device, 
                 save_model=False, model_path=None, earlyStop=False, save_path=None, 
                 lr=1e-4, weight_decay=1e-2, score_thre=0.5, evl_name='auc_score'):
        
        self.model = model
        self.device = device
        self.epochs = epochs
        self.score_thre = score_thre
        self.save_model = save_model
        self.earlyStop = earlyStop
        if save_model:
            self.model_path = model_path
        if earlyStop:
            self.early_stopping = EarlyStopping(save_path)
        self.evl_name = evl_name
        
        self.criterion = getattr(nn, criterion)()
        self.optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=lr,
                                                         weight_decay=weight_decay)
        self.scheduler = getattr(torch.optim.lr_scheduler, scheduler)(self.optimizer, 
                                                                      T_max=epochs,
                                                                      eta_min=lr/100)
        
        self.scale = nn.Sigmoid()
        self.metrics_dict = {'train_loss':[], 'val_loss':[], 
                             'acc':[], 'precision':[], 'recall':[], 'f1':[],
                             'ap':[], 'mcc':[], 'auc_score':[], 'prc_score':[]}
        
    
    def model_train(self, train_set, val_set):

        for epoch in range(self.epochs):
            self.model.train()
            trian_pbar = BackgroundGenerator(train_set)
            train_loss = self.train_epoch(trian_pbar)
            train_loss = train_loss / len(train_set)
            
            self.model_eval(val_set)
            self.scheduler.step()
            for loss_name, loss in zip(['train_loss'], 
                                       [train_loss]):
                v = self.metrics_dict[loss_name]
                v.append(loss)
                
                self.metrics_dict.update({loss_name : v})
            
            metric_str = ''
            for name, metric in self.metrics_dict.items():
                metric_str += f'{name}: {round(metric[-1],4)}; '
            print(f'epoch {epoch}:\n{metric_str}')
            
            if self.earlyStop:
                self.early_stopping(self.metrics_dict['val_loss'][-1], self.model, epoch)
                if self.early_stopping.early_stop:
                    print(f"Early stopping in epoch {epoch}")
                    return pd.DataFrame(self.metrics_dict, columns=list(self.metrics_dict.keys()))
                    break
            
            if epoch == 0:
                best_score = self.metrics_dict[self.evl_name][-1]
                best_model = copy.deepcopy(self.model)
                best_epoch = 0
            else:
                if self.metrics_dict[self.evl_name][-1] > best_score:
                    best_score = self.metrics_dict[self.evl_name][-1]
                    best_model = copy.deepcopy(self.model)
                    best_epoch = epoch
            
        if self.save_model:
            torch.save(best_model.state_dict(), os.path.join(self.model_path, 'best_model.pth'))
            print('The metrics of best model are:')
            metric_str = f'Best epoch: {best_epoch}'
            for name, metric in self.metrics_dict.items():
                metric_str += f'{name}: {round(metric[best_epoch],4)}; '
            print(metric_str)
            
        return pd.DataFrame(self.metrics_dict, columns=list(self.metrics_dict.keys()))


    def train_epoch(self, trian_pbar):
        train_loss = 0

        for prot_repr, prot_feat, lig_repr, lig_feat, prot_mask, lig_mask, diazo, alkyne, label in tqdm(trian_pbar):

            prot_repr = prot_repr.to(self.device)
            prot_feat = prot_feat.to(self.device)
            lig_repr = lig_repr.to(self.device)
            lig_feat = lig_feat.to(self.device)
            prot_mask = prot_mask.to(self.device)
            lig_mask = lig_mask.to(self.device)
            diazo = diazo.to(self.device)
            alkyne = alkyne.to(self.device)
            label = label.to(self.device)
            
            prot_repr = prot_repr.float()
            self.optimizer.zero_grad()
            
            out, _, _, _ = self.model(
                                    #Protein information
                                    prot_repr, prot_feat, prot_mask,
                                    #Ligand information
                                    lig_repr, lig_feat, lig_mask
                                    )
            
            loss = self.criterion(out.squeeze(), label.float())
            train_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
        
        return train_loss


    def model_eval(self, val_set):
        self.model.eval()
        val_loss = 0
        y_true, y_pred, y_prob = [], [], []
        val_bar = BackgroundGenerator(val_set)
        
        with torch.no_grad():
            for prot_repr, prot_feat, lig_repr, lig_feat, prot_mask, lig_mask, diazo, alkyne, label in val_bar:

                prot_repr = prot_repr.to(self.device)
                prot_feat = prot_feat.to(self.device)
                lig_repr = lig_repr.to(self.device)
                lig_feat = lig_feat.to(self.device)
                prot_mask = prot_mask.to(self.device)
                lig_mask = lig_mask.to(self.device)
                diazo = diazo.to(self.device)
                alkyne = alkyne.to(self.device)
                label = label.to(self.device)
                
                prot_repr = prot_repr.float()
                
                out, _, _, _ = self.model(
                                    #Protein information
                                    prot_repr, prot_feat, prot_mask,
                                    #Ligand information
                                    lig_repr, lig_feat, lig_mask
                                    )
                
                score = out.squeeze()
                loss = self.criterion(score.squeeze(), label.float())
                val_loss += loss.item()
                
                prob = self.scale(score)
                pred = (prob >= self.score_thre).int()
                y_true.extend(label.tolist())
                y_pred.extend(pred.tolist())
                y_prob.extend(prob.tolist())
        
            val_loss = val_loss / len(val_set)
            self.metrics_calculation(y_true, y_pred, y_prob)
            
            for loss_name, loss in zip(['val_loss'], 
                                       [val_loss]):
                
                v = self.metrics_dict[loss_name]
                v.append(loss)
                self.metrics_dict.update({loss_name : v})
    

    def metrics_calculation(self, y_true, y_pred, y_prob, out_format=None):
        y_true, y_pred, y_prob = np.float64(y_true), np.float64(y_pred), np.float64(y_prob)

        auc_score = roc_auc_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None)[1]
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        ap = average_precision_score(y_true, y_prob,average=None)
        tpr, fpr, _ = precision_recall_curve(y_true, y_prob)
        prc_score = auc(fpr, tpr)
        
        score_dict = {
            'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 
            'ap': ap, 'mcc': mcc, 'auc_score': auc_score, 'prc_score': prc_score
                      }
        
        for key in self.metrics_dict.keys():
            if key in score_dict.keys():
                v = self.metrics_dict[key]
                v.append(score_dict[key])
                self.metrics_dict.update({key : v})

        if out_format == 'frame':
            metrics = pd.DataFrame([score_dict]).T.reset_index(drop=False)
            metrics.columns = ['metrics name','values']
            return metrics

    def model_test(self, test_set, model_name='best_model.pth'):
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, model_name)))
        self.model.eval()
        
        out_lt, label_lt = np.array([]), np.array([])

        for prot_repr, prot_feat, lig_repr, lig_feat, prot_mask, lig_mask, diazo, alkyne, label in test_set:
            prot_repr = prot_repr.to(self.device)
            prot_feat = prot_feat.to(self.device)
            lig_repr = lig_repr.to(self.device)
            lig_feat = lig_feat.to(self.device)
            prot_mask = prot_mask.to(self.device)
            lig_mask = lig_mask.to(self.device)
            diazo = diazo.to(self.device)
            alkyne = alkyne.to(self.device)
            label = label.to(self.device)
            
            prot_repr = prot_repr.float()
            
            out, _, _, _ = self.model(
                                    #Protein information
                                    prot_repr, prot_feat, prot_mask,
                                    #Ligand information
                                    lig_repr, lig_feat, lig_mask,
                                    )
            
            prob = self.scale(out).tolist()
            out_lt = np.append(out_lt, prob)
            label_lt = np.append(label_lt, label.tolist())
            
        return out_lt, label_lt
    
    
    
