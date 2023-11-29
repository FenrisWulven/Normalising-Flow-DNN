import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.distributions import Dirichlet
from sklearn import metrics

def accuracy(y, preds): 
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
    correct_preds = (preds == y).sum().item().cpu().detach().numpy()

    if uncertainty_type == 'epistemic':
        scores = alpha.max(-1)[0].cpu().detach().numpy() #max_alpha_values
    elif uncertainty_type == 'aleatoric':
        p = F.normalize(alpha, p=1, dim=-1) # make probabilities from alpha
        scores = p.max(-1)[0].cpu().detach().numpy() #max probability values

    if score_type == 'AUROC':
        false_pr, true_pr, thresholds = metrics.roc_curve(correct_preds, scores) #fpr is false positive rate, tpr is true positive rate
        return metrics.auc(false_pr, true_pr)
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

    correct_preds = np.concatenate([np.ones(batch_size), np.zeros(ood_batch_size)], axis=0) # what does this do? It     
    scores = np.concatenate([scores, ood_scores], axis=0)

    if score_type == 'AUROC':
        false_pr, true_pr, thresholds = metrics.roc_curve(correct_preds, scores)
        return metrics.auc(false_pr, true_pr)
    elif score_type == 'APR':
        return metrics.average_precision_score(correct_preds, scores)
    else:
        raise NotImplementedError


def entropy(alpha, uncertainty_type, n_bins=10, plot=True):
    entropy = []

    if uncertainty_type == 'aleatoric':
        p = F.normalize(alpha, p=1, dim=-1)
        entropy.append(Categorical(p).entropy().squeeze().cpu().detach().numpy())
    elif uncertainty_type == 'epistemic':
        entropy.append(Dirichlet(alpha).entropy().squeeze().cpu().detach().numpy())

    if plot:
        plt.hist(entropy, n_bins)
        plt.show()
    return entropy



output_dim = 10
reg = 5e-5
batch_size = 3
alpha = torch.tensor([[2.4624, 3.1070, 2.6662, 3.1353, 1.8208, 2.6505, 2.5713, 2.9230, 2.3965,
         2.5298],
        [2.0680, 2.1061, 2.3548, 2.0821, 1.7339, 2.1956, 2.0966, 2.2842, 1.9681,
         1.9354],
        [3.8852, 9.1216, 5.0135, 7.3253, 2.2495, 4.5210, 9.1264, 6.1262, 3.6266,
         3.8111]])
preds = torch.max(alpha, dim=-1)[1]
y_hot = torch.tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
y = torch.tensor([2, 0, 0])

def loss_postnet(alpha, y, batch_size): #UCE loss 
    #alpha is the p vector with alphas for each class of size: [batch x num_classes]
    #y is the ground-truth class labels of size: [batch x 1]

    #number_classes = alpha.size(1)
    
    # print("number classes",number_classes)
    # print("alpha",alpha)
    # print("alpha_c",alpha[range(batch_size),y])
    alpha_0 = torch.sum(alpha, dim=1, keepdim=False) #[batch]
    digamma_alpha_0 = torch.digamma(alpha_0) # batch x 1, hver obs får sin egent logprobs
    digamma_alpha_c = torch.digamma(alpha[range(batch_size),y]) # [batch] 
    # IMPORTANT: this gets the alpha value for the correct class for each obs in batch, instead of all alpha values for each obs

    # print("digamma 0",digamma_alpha_0) # digamma 0 values are the same as in the paper - just NOT repeated 10 times
    # print("digamma alpha for correct class",digamma_alpha_c)
    # we are only interested in optimising this value, and the other values are irrelevant?
    # print("y",y)
    # print("range batchsize",range(batch_size))
    # print("indexes",[range(batch_size),y])
    uce_loss_elements = digamma_alpha_0 - digamma_alpha_c #elementwise produces negative values
    entropy_reg = Dirichlet(alpha).entropy() #tensor of batch shape
    #print("uce loss elements",uce_loss_elements)
    #print("mean uce loss", torch.sum(uce_loss_elements))
    #print("entropy reg",- reg * torch.sum(entropy_reg))
    postnet_loss = torch.sum(uce_loss_elements) - reg * torch.sum(entropy_reg) #negative since we want to minimize the loss
    return postnet_loss

def UCE_loss(alpha, soft_output):
    #print("Alpha: ", alpha.shape, alpha) 
    # torch.Size([3, 10]) with first alpha is tensor([[  4.6238, 100.8251,   8.1585,  54.2917,   1.6244,  10.2861,  30.9715, 19.3688,   4.1660,   4.4290]
    #print("Soft output: ", soft_output.shape, soft_output)
    # torch.Size([3, 10]) with first y_hot tensor([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
    alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, output_dim)
    #print("\nPAPER alpha_0", alpha_0.shape, alpha_0)
    #print("PAPER alpha", alpha.shape, alpha)
    # print("PAPER digamma 0", torch.digamma(alpha_0)) 
    # print("PAPER digamma alpha", torch.digamma(alpha))
    entropy_reg = Dirichlet(alpha).entropy()
    #print("PAPER uce loss elements multplied by y_hot", soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha)) )
    #print("PAPER sum uce loss", torch.sum(soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))) )
    #print("PAPER entropy_reg", - reg * torch.sum(entropy_reg))
    UCE_loss = torch.sum(soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))) - reg * torch.sum(entropy_reg)
    return UCE_loss

#print("UCE loss:", loss_postnet(alpha, y, batch_size), "\n")
#print("UCE loss2:", UCE_loss(alpha, y_hot))
