import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.distributions import Dirichlet
from sklearn import metrics

import os
get_cwd = os.getcwd()
root_dir = get_cwd
plots_dir = os.path.join(root_dir, 'saved_plots')


def accuracy(y, alpha): 
    preds = torch.max(alpha, dim=-1)[1] # get predicated class label for each sample in batch
    accuracy = metrics.accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
    return accuracy

def brier_score(y, alpha):
    # y is the ground-truth class labels of size: [batch]
    # alpha is the p vector with alphas for each class of size: [batch x num_classes]
    batch_size = alpha.size(0)
    # Normalize alpha to get probabilities
    p = F.normalize(alpha, p=1, dim=-1)
    p[torch.arange(batch_size), y] -= 1 # subtract 1 from the correct class for each obs in batch
    p_normed = p.norm(dim=-1) # get the quatric (frobenius) norm of p as||p_bar^(i) - y^(i)||_2
    brier_score = p_normed.mean().cpu().detach().numpy() #1/N sum^N_i 
    return brier_score

def confidence(y, alpha, score_type='AUROC', uncertainty_type='aleatoric'):
    preds = torch.max(alpha, dim=-1)[1]
    correct_preds = (preds == y)
    if uncertainty_type == 'epistemic':
        scores = alpha.max(-1)[0].cpu().detach().numpy() #max_alpha_values
       
    elif uncertainty_type == 'aleatoric':
        p = F.normalize(alpha, p=1, dim=-1) # make probabilities from alpha
        scores = p.max(-1)[0].cpu().detach().numpy() #max probability values

    if score_type == 'AUROC':
        if all(correct_preds):
            print("Perfect classification achieved, AUROC is not applicable.")
            return 1.0 # Treated as 1.0 insead of np.nan
        try: 
            false_pr, true_pr, thresholds = metrics.roc_curve(correct_preds, scores) #fpr is false positive rate, tpr is true positive rate
            return metrics.auc(false_pr, true_pr)
        except ValueError:
            print('Error calculating AUROC')
            return np.nan
            
    elif score_type == 'APR':
        return metrics.average_precision_score(correct_preds, scores)
    else:
        raise NotImplementedError



def ood_detection(alpha, ood_alpha, score_type='AUROC', uncertainty_type='aleatoric'):
    batch_size = alpha.size(0) 
    ood_batch_size = ood_alpha.size(0)
    if uncertainty_type == 'epistemic':
        scores = alpha.sum(-1).cpu().detach().numpy()
        ood_scores = ood_alpha.sum(-1).cpu().detach().numpy()

    elif uncertainty_type == 'aleatoric':
        p = F.normalize(alpha, p=1, dim=-1)
        scores = p.max(-1)[0].cpu().detach().numpy()
        ood_p = F.normalize(ood_alpha, p=1, dim=-1)
        ood_scores = ood_p.max(-1)[0].cpu().detach().numpy()

    correct_preds = np.concatenate([np.ones(batch_size), np.zeros(ood_batch_size)], axis=0) #correct_preds are 1 for in-distribution and 0 for ood
    scores = np.concatenate([scores, ood_scores], axis=0)

    if score_type == 'AUROC':
        false_pr, true_pr, thresholds = metrics.roc_curve(correct_preds, scores)
        return metrics.auc(false_pr, true_pr)
    elif score_type == 'APR':
        return metrics.average_precision_score(correct_preds, scores)
    else:
        raise NotImplementedError


def entropy(alpha, uncertainty_type): #dataset_name, model_name, n_bins=10, plot=True
    entropy = []

    if uncertainty_type == 'aleatoric':
        p = F.normalize(alpha, p=1, dim=-1)
        entropy.append(Categorical(p).entropy().squeeze().cpu().detach().numpy())
    elif uncertainty_type == 'epistemic':
        entropy.append(Dirichlet(alpha).entropy().squeeze().cpu().detach().numpy())

    # if plot:
    #     plt.figure(figsize=(8, 6))
    #     plt.hist(entropy, n_bins, density=True, label=f'Entropy - {uncertainty_type.capitalize()}',  )
    #     plt.xlabel('Entropy')
    #     plt.ylabel('Density')
    #     plt.title(f'Entropy Histogram - {uncertainty_type.capitalize()} Uncertainty')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(plots_dir, dataset_name, 'entropy' + uncertainty_type + model_name + '.png'), bbox_inches='tight')    
    return entropy


