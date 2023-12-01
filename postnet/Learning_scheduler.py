import torch
import torch.nn as nn
import torch.nn.utils as utils
import numpy as np
import math
import wandb
import os

from metrics import accuracy

get_cwd = os.getcwd()
root_dir = get_cwd
print(root_dir)
#root_dir = os.path.dirname(get_cwd)
models_dir = os.path.join(root_dir, 'saved_models')

class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimiser, warmup_steps, start_lr, end_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_increment = (end_lr - start_lr) / warmup_steps
        super().__init__(optimiser, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = self.start_lr + self.last_epoch * self.lr_increment
            return [lr for _ in self.base_lrs]
        return self.base_lrs


def init_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        nn.init.zeros_(model.bias) 


def train(model, optimiser, train_loader, val_loader, num_epochs, validation_every_steps, early_stop_delta, 
          early_stop_patience, warmup_scheduler, training_scheduler, warmup_steps, N_counts, set_lengths, device, save_model):
    model.train()
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    all_train_losses = []
    best_val_loss = float("Inf")
    step = 0 # how many batches we have trained on (each batch is 64 samples) #9000 training samples / 64 batch size = 140 batches per epoch
    counter = 0 # for early stopping 
    early_stopping = False
    total_steps_per_epoch = len(train_loader)  # Total batches (steps) per epoch
    warmup_epochs = math.ceil(warmup_steps / total_steps_per_epoch)  # Total warmup epochs
    print("Steps_per_epoch", total_steps_per_epoch)
    print("Warmup epochs:", warmup_epochs)
    wandb.watch(model, log="all")

    for epoch in range(num_epochs): #epoch is one forward pass through the entire training set
        train_losses_batches, train_accuracies_batches = [], []
        #batches_counter = 0

        for batch_index, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to(device), y_train.to(device)
            # batches_counter += 1

            # Forward pass
            alpha = model(X_train, N_counts['train'])
            loss = model.loss_postnet(alpha, y_train) #batch size
            # Perform one training step
            optimiser.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimiser.step()
            if step < warmup_steps:
                warmup_scheduler.step()
            
            step += 1
            #train_losses.append(loss.item())

            # Compute training accuracy and loss for this batch
            with torch.no_grad():
                #preds = torch.max(alpha, dim=-1)[1]
                train_accuracy_batch = accuracy(y_train, alpha)
                train_accuracies_batches.append(train_accuracy_batch)
                train_losses_batches.append(loss.item())
                all_train_losses.append(loss.item())
                current_lr = optimiser.param_groups[0]['lr']
                wandb.log({"batch_train_losses": loss.item(), "batch_train_accuracy": 
                           train_accuracy_batch, "step": step, "learning_rate": current_lr, "epoch": epoch})
                #wandb.log({"batch_train_losses": loss.item(), "batch_train_accuracy": 
                           #train_accuracy_batch, "step": step})
                
                #train_accuracies.append(batch_accuracy)

            # if epoch >= warmup_epochs:
            #     if (epoch - warmup_epochs) % annealing_interval == 0:
            #         training_scheduler.step()

            if step % validation_every_steps == 0:
                train_loss = np.mean(train_losses_batches)
                train_losses.append(train_loss)
                train_accuracy = np.mean(train_accuracies_batches)
                train_accuracies.append(train_accuracy)
                wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "step": step})

                val_losses_batches = []
                #val_accuracies_batches = []
                val_correct = []
                model.eval()
                with torch.no_grad():   
                    for batch_index, (X_val, y_val) in enumerate(val_loader):
                        X_val, y_val = X_val.to(device), y_val.to(device)
                        # Evaluation Forward pass
                        alpha = model(X_val, N_counts['val']) # gives a vector with alphas for each class
                        loss = model.loss_postnet(alpha, y_val) #gives a loss
                        
                        # Evaluation accuracy and loss for this batch
                        preds = torch.max(alpha, dim=-1)[1]
                        
                        correct_batch = (preds == y_val).sum().item()
                        val_correct.append(correct_batch)

                        #Maybe: Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                        #val_accuracies_batches.append(accuracy(y_val, preds) * len(X_val))

                        # append the loss for this batch
                        val_losses_batches.append(loss.item())

                val_accuracy = sum(val_correct) / set_lengths['val'] 
                 #The final batch of DataLoader may be smaller (drop_last=False).
                val_accuracies.append(val_accuracy)
                val_loss = np.mean(val_losses_batches) 
                val_losses.append(val_loss)
                wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "step": step})
                print(f"Step: {step}, Epoch: {epoch+1}\tTrain Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")
                #### Lave plots med meshgrid f-funktion af normalising flow undervejs for at se ændringen

                model.train()

                if val_losses[-1] < -1.:
                    print("Unstable training")
                    break
                if np.isnan(val_losses[-1]):
                    print('Detected NaN Loss')
                    break
                # If val_loss is the best so far, save the model state_dict and reset the early stopping counter
                if val_losses[-1] < best_val_loss:
                    best_val_loss = val_losses[-1]
                    counter = 0
                    best_model = model.state_dict()
                    torch.save({'epoch': epoch, 'model_state_dict': best_model, 'loss': best_val_loss}, os.path.join(models_dir, save_model))
                    print('Model saved')

                # Early stopping - if val_loss is not improving (plus a delta e-4 as buffer) then start counter
                # after patience of a certain number of validations, then stop training
                elif val_losses[-1] > (best_val_loss + early_stop_delta):
                    counter += 1
                    if counter >= early_stop_patience:
                        #print("Early stopping")
                        early_stopping = True
                        break
        
        # Update training scheduler (annealing LR)
        if epoch >= warmup_epochs:
                #if (epoch - warmup_epochs) % annealing_interval == 0:
                training_scheduler.step()
        #current_lr = optimiser.param_groups[0]['lr']
        #print(f"Epoch {epoch}: Current Epoch LR = {current_lr}")

        if early_stopping: # if true
            print("Early stopping triggered. Exiting training.")
            break  # Break out of the outer loop
    print("Finished training.")
    return train_losses, val_losses, train_accuracies, val_accuracies, all_train_losses #,model
