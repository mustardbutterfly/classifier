import torch
import torch.nn.functional as F


def kl_divergence_loss(preds, weak_pairs, lambda_entropy=0.1):
    loss = 0.0
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

    # Convert logits to probability distributions
    preds_log = F.log_softmax(preds, dim=1)  # Log-probabilities
    preds_prob = F.softmax(preds, dim=1)  # Probabilities

    # KL divergence for weakly supervised pairs
    for i, j in weak_pairs:
        loss += kl_loss(preds_log[i], preds_prob[j])  # D_KL(preds[i] || preds[j])
        loss += kl_loss(preds_log[j], preds_prob[i])  # D_KL(preds[j] || preds[i]) (symmetric)

    # Normalize by the number of pairs
    if len(weak_pairs) > 0:
        loss /= len(weak_pairs)

    # Entropy regularization to prevent model collapse
    entropy_loss = -torch.sum(preds_prob * preds_log, dim=1).mean()

    return loss + lambda_entropy * entropy_loss

def kl_loss_between_pair(input1, input2):
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

    # Convert logits to probability distributions
    preds_input1 = F.log_softmax(input1, dim=1)  # Log-probabilities
    preds_input2 = F.softmax(input2, dim=1)  # Probabilities

    return kl_loss(preds_input1, preds_input2)