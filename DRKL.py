import torch
import torch.nn.functional as F

def drkl_loss(logits_mlp, logits_gnn, alpha, beta, temperature=1.0):

    l_gnn = logits_gnn / temperature
    l_mlp = logits_mlp / temperature
    
    Q_mlp = F.softmax(l_mlp, dim=-1)
    
    loss = (alpha * F.cross_entropy(l_gnn, Q_mlp, reduction='mean') -
            beta  * F.cross_entropy(l_mlp, Q_mlp, reduction='mean')) * (temperature ** 2)
    
    return loss
