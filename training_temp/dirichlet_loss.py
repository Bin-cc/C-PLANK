import torch
import torch.nn as nn
import torch.nn.functional as F


class DirichletLossWithAnneal(nn.Module):
    
    def __init__(self, 
                 num_classes=2,
                 lam_target=0.1, 
                 lambda_rank_target=0.3, 
                 grade_weights=[0.9, 1.0, 1.1],
                 lam_warmup_epochs=10,
                 rank_warmup_epochs=10,
                 pos_reg_weight=0.01,
                 margin = 0.02):
        
        super().__init__()
        self.num_classes = num_classes
        self.lam_target = lam_target
        self.lambda_rank_target = lambda_rank_target
        self.grade_weights = grade_weights
        
        self.lam_warmup_epochs = lam_warmup_epochs
        self.rank_warmup_epochs = rank_warmup_epochs
        
        self.current_lam = 0.0
        self.current_lambda_rank = 0.0
        
        self.pos_reg_weight = pos_reg_weight
        self.margin = margin
    
    @staticmethod
    def _kl_divergence(alpha):
        beta = torch.ones_like(alpha)
        S_alpha = torch.sum(alpha, dim=-1, keepdim=True)
        S_beta = torch.sum(beta, dim=-1, keepdim=True)
        
        ln_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
        ln_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)
        
        dg_alpha = torch.digamma(alpha)
        dg_S_alpha = torch.digamma(S_alpha)
        
        kl = ln_alpha + ln_beta + torch.sum((alpha - beta)*(dg_alpha - dg_S_alpha), dim=-1, keepdim=True)
        return kl
    
    
    def step(self, epoch):
        if epoch < self.lam_warmup_epochs:
            self.current_lam = self.lam_target * (epoch / self.lam_warmup_epochs)
        else:
            self.current_lam = self.lam_target
        
        if epoch < self.rank_warmup_epochs:
            ratio = epoch / self.rank_warmup_epochs
            self.current_lambda_rank = self.lambda_rank_target * ratio
        else:
            self.current_lambda_rank = self.lambda_rank_target
    
    
    def forward(self, alphas, y, grade_labels):
        device = alphas.device
        y_one_hot = F.one_hot(y.long(), num_classes=self.num_classes).float()
        
        alphas = alphas.reshape(-1, self.num_classes)
        S = torch.sum(alphas, dim=-1, keepdim=True)
        p = alphas / S
        
        A = torch.sum(torch.pow((y_one_hot - p), 2), dim=-1, keepdim=True)
        B = torch.sum((p*(1 - p)) / (S + 1), dim=-1, keepdim=True)
        SOS = (A + B).squeeze(-1)

        weight_tensor = torch.ones_like(grade_labels, dtype=torch.float32, device=device)
        weight_tensor[grade_labels == 2] = self.grade_weights[2]
        weight_tensor[grade_labels == 1] = self.grade_weights[1]
        weight_tensor[grade_labels == 0] = self.grade_weights[0]
        
        weighted_SOS = SOS * weight_tensor
        
        alpha_hat = y_one_hot + (1 - y_one_hot) * alphas
        KL_loss = self.current_lam * self._kl_divergence(alpha_hat).squeeze(-1)
    
        base_loss = weighted_SOS + KL_loss
        base_loss = torch.mean(base_loss, dim=-1)
        
        S_flat = S.squeeze(-1)
        u = self.num_classes / S_flat
        
        if self.current_lambda_rank > 1e-8:
            high_mask = (grade_labels == 2)
            medium_mask = (grade_labels == 1)
            low_mask = (grade_labels == 0)
            
            if high_mask.any() and medium_mask.any():
                hm_loss = F.relu(u[high_mask].mean() - u[medium_mask].mean() + self.margin)
            
            if medium_mask.any() and low_mask.any():
                ml_loss = F.relu(u[medium_mask].mean() - u[low_mask].mean() + self.margin)
        
        total_loss = base_loss + self.current_lambda_rank * (hm_loss + ml_loss)
        
        return total_loss