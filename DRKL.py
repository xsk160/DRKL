import torch
import torch.nn.functional as F

def drkl_loss(logits_mlp, logits_gnn, alpha, beta, temperature):

    loss_CFT = F.cross_entropy(logits_gnn, logits_mlp)
    loss_RT = F.cross_entropy(logits_mlp, logits_mlp)

    loss = alpha*loss_CFT - beta*loss_RT

    return loss

