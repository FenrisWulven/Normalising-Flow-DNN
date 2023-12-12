import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
import sys
import os
get_cwd = os.getcwd()
root_dir = get_cwd
plots_dir = os.path.join(root_dir, 'saved_plots')
sys.path.append(root_dir) 

from postnet.metrics import accuracy, brier_score, confidence, ood_detection, entropy


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

def evaluate_model(model, test_loader, ood_dataset_loaders, N_test, ood_N, dataset_name, model_name, device):
    save_model = os.path.join(root_dir, 'saved_models', dataset_name, model_name + '.pth')
    checkpoint = torch.load(save_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_y_test, all_alphas = compute_all_y_alphas(model, test_loader, N_test, device)

    entropy_data = {}
    test_metrics = {}
    test_metrics['accuracy'] = accuracy(y=all_y_test, alpha=all_alphas)
    test_metrics['brier_score'] = brier_score(y=all_y_test, alpha=all_alphas)
    test_metrics['confidence_APR_aleatoric'] = confidence(y=all_y_test, alpha=all_alphas, score_type='APR', uncertainty_type='aleatoric')
    test_metrics['confidence_APR_epistemic'] = confidence(y=all_y_test, alpha=all_alphas, score_type='APR', uncertainty_type='epistemic')
    test_metrics['confidence_AUROC_aleatoric'] = confidence(y=all_y_test, alpha=all_alphas, score_type='AUROC', uncertainty_type='aleatoric')
    test_metrics['confidence_AUROC_epistemic'] = confidence(y=all_y_test, alpha=all_alphas, score_type='AUROC', uncertainty_type='epistemic')
    entropy_data[f'entropy_aleatoric_{dataset_name}'] = entropy(alpha=all_alphas, uncertainty_type='aleatoric')
    entropy_data[f'entropy_epistemic_{dataset_name}'] = entropy(alpha=all_alphas, uncertainty_type='epistemic')

    # Compute anomaly detection metrics for Fashion-MNIST and KMNIST in ood_dataset_loaders
    for ood_dataset_name, ood_dataset_loader in ood_dataset_loaders.items():
        _, ood_all_alphas = compute_all_y_alphas(model, ood_dataset_loader, ood_N[ood_dataset_name], device)
        test_metrics[f'ood_detection_APR_aleatoric_{ood_dataset_name}'] = ood_detection(alpha=all_alphas, ood_alpha=ood_all_alphas, score_type='APR', uncertainty_type='aleatoric')
        test_metrics[f'ood_detection_APR_epistemic_{ood_dataset_name}'] = ood_detection(alpha=all_alphas, ood_alpha=ood_all_alphas, score_type='APR', uncertainty_type='epistemic') 
        test_metrics[f'ood_detection_AUROC_aleatoric_{ood_dataset_name}'] = ood_detection(alpha=all_alphas, ood_alpha=ood_all_alphas, score_type='AUROC', uncertainty_type='aleatoric')
        test_metrics[f'ood_detection_AUROC_epistemic_{ood_dataset_name}'] = ood_detection(alpha=all_alphas, ood_alpha=ood_all_alphas, score_type='AUROC', uncertainty_type='epistemic')
        entropy_data[f'entropy_aleatoric_{ood_dataset_name}'] = entropy(alpha=ood_all_alphas, uncertainty_type='aleatoric')
        entropy_data[f'entropy_epistemic_{ood_dataset_name}'] = entropy(alpha=ood_all_alphas, uncertainty_type='epistemic')

    test_metrics_df = pd.DataFrame([test_metrics])
    csv_save_path = os.path.join(plots_dir, dataset_name, 'csv' + model_name + '.csv')
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    test_metrics_df.to_csv(csv_save_path, index=False)
    
    entropy_data_df = pd.DataFrame([entropy_data])
    csv_save_path = os.path.join(plots_dir, dataset_name, 'entropy_csv' + model_name + '.csv')
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    entropy_data_df.to_csv(csv_save_path, index=False)

    wandb.log(test_metrics)
    wandb.log(entropy_data)
    return test_metrics, entropy_data

def image_show(img):
    img = img.detach().cpu()
    img = img / 2 + 0.5   #Unnormalise
    with sns.axes_style("white"):
        plt.figure(figsize=(8, 8))
        plt.imshow(img.permute((1, 2, 0)).numpy())
        plt.axis('off')
        plt.show()


def plot_loss_acc(train_losses, val_losses, train_accuracies, val_accuracies, all_train_losses, dataset_name, model_name):
    # Make x-axis from validation every 50 steps, to just show total step
    validation_every_steps = 50
    training_steps = list(range(0, len(train_losses) * validation_every_steps, validation_every_steps))
    validation_steps = list(range(0, len(val_losses) * validation_every_steps, validation_every_steps))

    # Plot loss of training and validation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(training_steps, train_losses, label='Train Loss')
    axes[0].plot(validation_steps, val_losses, label='Validation Loss')
    axes[0].set_xlabel('Steps (validated every 50 steps)')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train and Validation Loss')
    axes[0].legend()

    # Plot accuracies of training and validation
    axes[1].plot(training_steps, train_accuracies, label='Train Accuracy')
    axes[1].plot(validation_steps, val_accuracies, label='Validation Accuracy')
    axes[1].set_xlabel('Steps (validated every 50 steps)')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Train and Validation Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, dataset_name, 'loss_acc' + model_name + '.png'), bbox_inches='tight') 
    plt.close()   

    # # plot all_train_losses
    # plt.figure(figsize=(12,8))
    # plt.plot(all_train_losses,  '.',label='Train Loss', alpha=0.3)
    # #plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Steps')
    # plt.ylabel('Loss')
    # plt.title('Train Loss')
    # plt.legend()
    # plt.savefig(os.path.join(plots_dir, save_path), bbox_inches='tight')

def plot_entropy(entropy_data, dataset_name, model_name, n_bins=30):
    aleatoric_data = {label: values for label, values in entropy_data.items() if 'aleatoric' in label}
    epistemic_data = {label: values for label, values in entropy_data.items() if 'epistemic' in label}

    def plot_uncertainty(data, uncertainty_type):
        plt.figure(figsize=(10, 6))
        for label, values in data.items():
            #finite_values = values[np.isfinite(values)] # Filter out non-finite values (aka. infinite values)
            #if len(finite_values) > 0:
            plt.hist(values, bins=n_bins, alpha=0.5, label=f'{label.replace("_", " ").capitalize()}') #label=f'{label.capitalize()} Entropy')
            #else: 
            #    print(f"No finite values to plot for {label}")
        plt.xlabel('Entropy')
        plt.ylabel('Density')
        plt.title(f'{uncertainty_type.capitalize()} Uncertainty Histogram for {dataset_name}')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(plots_dir, dataset_name, uncertainty_type + '_combined_entropy_' + model_name + '.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()  
    
    plot_uncertainty(aleatoric_data, 'aleatoric')
    plot_uncertainty(epistemic_data, 'epistemic')