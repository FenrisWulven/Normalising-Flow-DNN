import torch
import matplotlib.pyplot as plt
import seaborn as sns
#import sys
#sys.path.append('../')
#sys.path.append(r'c:\Users\ollie\OneDrive - Danmarks Tekniske Universitet\Uni\Bachelor Projekt\Normalising-Flow-DNN')
#from postnet.metrics import accuracy, brier_score, confidence, ood_detection, entropy
from metrics import accuracy, brier_score, confidence, ood_detection, entropy


def compute_all_y_alphas(model, loader, N, device):
    all_y_test = []
    all_alphas = []
    with torch.no_grad():
        for X_test, y_test in loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            # Forward pass
            alpha = model(X_test, N)
            all_y_test.append(y_test.cpu())
            all_alphas.append(alpha.cpu())
    
    all_y_test = torch.cat(all_y_test)
    all_alphas = torch.cat(all_alphas)

    return all_y_test, all_alphas

def evaluate_model(model, test_loader, ood_dataset_loaders, N_test, ood_N, checkpoint, device):
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    #test_loader.dataset
    all_y_test, all_alphas = compute_all_y_alphas(model, test_loader, N_test, device)

    test_metrics = {}
    test_metrics['accuracy'] = accuracy(y=all_y_test, alpha=all_alphas)
    test_metrics['brier_score'] = brier_score(y=all_y_test, alpha=all_alphas)
    test_metrics['confidence_APR_aleatoric'] = confidence(y=all_y_test, alpha=all_alphas, score_type='APR', uncertainty_type='aleatoric')
    test_metrics['confidence_APR_epistemic'] = confidence(y=all_y_test, alpha=all_alphas, score_type='APR', uncertainty_type='epistemic')
    
    # Compute anomaly detection metrics for Fashion-MNIST and KMNIST in ood_dataset_loaders
    for ood_dataset_name, ood_dataset_loader in ood_dataset_loaders.items():
        _, ood_all_alphas = compute_all_y_alphas(model, ood_dataset_loader, ood_N[ood_dataset_name], device)
        test_metrics[f'ood_detection_aleatoric_{ood_dataset_name}'] = ood_detection(alpha=all_alphas, ood_alpha=ood_all_alphas, score_type='APR', uncertainty_type='aleatoric')
        test_metrics[f'ood_detection_epistemic_{ood_dataset_name}'] = ood_detection(alpha=all_alphas, ood_alpha=ood_all_alphas, score_type='APR', uncertainty_type='epistemic') 

    return test_metrics

def image_show(img):
    img = img.detach().cpu()
    img = img / 2 + 0.5   #Unnormalise
    with sns.axes_style("white"):
        plt.figure(figsize=(8, 8))
        plt.imshow(img.permute((1, 2, 0)).numpy())
        plt.axis('off')
        plt.show()

