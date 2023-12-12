import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
import os
import sys
import numpy as np
get_cwd = os.getcwd()
root_dir = get_cwd
sys.path.append(root_dir)# 
from postnet.metrics import accuracy, brier_score, confidence, ood_detection, entropy
plots_dir = os.path.join(root_dir, 'saved_plots')

def compute_ensemble_all_y_avgprobs(ensemble_models, loader, device):
    batch_y_test = []
    ensemble_batch_probs = []
    with torch.no_grad():
        for X_test, y_test in loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            #all_x_test.append(X_test)
            batch_y_test.append(y_test.cpu()) # this is size [batch_size] with values 0-9 as the class labels

            batch_probs = []
            for model in ensemble_models:
                model.eval()
                output = model(X_test) #outputs logits
                probs = F.softmax(output, dim=1)
                batch_probs.append(probs.cpu()) # for each model this is its predictions for the batch of size [batch_size x num_classes]
        
            # Average probabilities across models for this batch
            batch_avg_probs = torch.stack(batch_probs).mean(dim=0)
            ensemble_batch_probs.append(batch_avg_probs)

    # Concatenate results from all batches
    all_y_test = torch.cat(batch_y_test)
    ensemble_avg_probs = torch.cat(ensemble_batch_probs)

    return all_y_test, ensemble_avg_probs


# Evaluate Ensemble
def evaluate_ensemble(ensemble_models, test_loader, ood_dataset_loaders, dataset_name, model_names, ensemble_name, device):
    # Load saved models
    for model, model_name in zip(ensemble_models, model_names):
        save_model = os.path.join(root_dir, 'saved_models', dataset_name, model_name + '.pth')
        checkpoint = torch.load(save_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    # Get a full ensemble avg prediction 
    all_y_test, ensemble_avg_probs = compute_ensemble_all_y_avgprobs(ensemble_models, test_loader, device)
    print("all_y_test", all_y_test.size())
    print("ensemble_avg_probs", ensemble_avg_probs.size())
    print("First ensemble_avg_probs", ensemble_avg_probs[0])

    entropy_data = {}
    test_metrics = {}
    test_metrics['accuracy'] = accuracy(y=all_y_test, alpha=ensemble_avg_probs)
    test_metrics['brier_score'] = brier_score(y=all_y_test, alpha=ensemble_avg_probs)
    test_metrics['confidence_APR_aleatoric'] = confidence(y=all_y_test, alpha=ensemble_avg_probs, score_type='APR', uncertainty_type='aleatoric')
    test_metrics['confidence_APR_epistemic'] = confidence(y=all_y_test, alpha=ensemble_avg_probs, score_type='APR', uncertainty_type='epistemic')
    test_metrics['confidence_AUROC_aleatoric'] = confidence(y=all_y_test, alpha=ensemble_avg_probs, score_type='AUROC', uncertainty_type='aleatoric')
    test_metrics['confidence_AUROC_epistemic'] = confidence(y=all_y_test, alpha=ensemble_avg_probs, score_type='AUROC', uncertainty_type='epistemic')
    entropy_data[f'entropy_aleatoric_{dataset_name}'] = entropy(alpha=ensemble_avg_probs, uncertainty_type='aleatoric')
    entropy_data[f'entropy_epistemic_{dataset_name}'] = entropy(alpha=ensemble_avg_probs, uncertainty_type='epistemic')

    # Compute anomaly detection metrics for Fashion-MNIST and KMNIST in ood_dataset_loaders
    for ood_dataset_name, ood_dataset_loader in ood_dataset_loaders.items():
        _, ood_ensemble_avg_probs = compute_ensemble_all_y_avgprobs(ensemble_models, ood_dataset_loader, device)
        test_metrics[f'ood_detection_APR_aleatoric_{ood_dataset_name}'] = ood_detection(alpha=ensemble_avg_probs, ood_alpha=ood_ensemble_avg_probs, score_type='APR', uncertainty_type='aleatoric')
        test_metrics[f'ood_detection_APR_epistemic_{ood_dataset_name}'] = ood_detection(alpha=ensemble_avg_probs, ood_alpha=ood_ensemble_avg_probs, score_type='APR', uncertainty_type='epistemic') 
        test_metrics[f'ood_detection_AUROC_aleatoric_{ood_dataset_name}'] = ood_detection(alpha=ensemble_avg_probs, ood_alpha=ood_ensemble_avg_probs, score_type='AUROC', uncertainty_type='aleatoric')
        test_metrics[f'ood_detection_AUROC_epistemic_{ood_dataset_name}'] = ood_detection(alpha=ensemble_avg_probs, ood_alpha=ood_ensemble_avg_probs, score_type='AUROC', uncertainty_type='epistemic')
        entropy_data[f'entropy_aleatoric_{ood_dataset_name}'] = entropy(alpha=ood_ensemble_avg_probs, uncertainty_type='aleatoric')
        entropy_data[f'entropy_epistemic_{ood_dataset_name}'] = entropy(alpha=ood_ensemble_avg_probs, uncertainty_type='epistemic')


    test_metrics_df = pd.DataFrame([test_metrics])
    csv_save_path = os.path.join(plots_dir, dataset_name, 'csv' + ensemble_name + '.csv')
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    test_metrics_df.to_csv(csv_save_path, index=False)
    
    # save entropy data
    entropy_data_df = pd.DataFrame([entropy_data])
    csv_save_path = os.path.join(plots_dir, dataset_name, 'entropy_csv' + ensemble_name + '.csv')
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    entropy_data_df.to_csv(csv_save_path, index=False)
    #wandb.log(test_metrics)
    #wandb.log(entropy_data)
    return test_metrics, entropy_data
