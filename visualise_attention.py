

import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
# skimage resize
from skimage.transform import resize

import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT
from imageclassification import set_seed, select_two_classes_from_cifar10, prepare_dataloaders#, visualize_attention

def visualize_attention(img_batch, attention_maps, patch_size=(4,4), channels=3, nrows=1, ncols=1, figsize=(10, 10)):
    if torch.is_tensor(img_batch):
        img_batch = img_batch.cpu().numpy()
    

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle('Attention Maps Visualised', fontdict={'fontname': 'Garamond', 'weight': 'bold'}, size=26)
    fig.subplots_adjust(top=0.95, hspace=0.5)
    for idx, ax in enumerate(axs.flat):
        if idx >= img_batch.shape[0]:
            break

        img = np.transpose(img_batch[idx], (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())
    
        # 16, 65, 65
        attention_map = attention_maps[idx].detach().cpu().numpy()
        #print("attention_map", attention_map.shape) # 16x65x65
        # first row - remove first element to get a 64x1 vector showing which is the attention weights for classification token
        # take only the first row of the attention map
        attention_map = attention_map[:, 0, 1:] 
        #print("attention_map row", attention_map.shape) # 16x64
        # Resize 4x64 to 4x8x8
        attention_map_resized = attention_map.reshape((attention_map.shape[0], 8, 8))
        #print("attention_map_resized", attention_map_resized.shape) # 4x8x8
        #Average the heads
        attention_map_resized = np.mean(attention_map_resized, axis=0)
        #print("attention_map_resized avg", attention_map_resized.shape) # 8x8
        # Bilieanr interpolation to scale to 32x32 (remember to squueze)
        attention_map_inter = F.interpolate(torch.tensor(attention_map_resized).unsqueeze(0).unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze().numpy()
        #print("attention_map_inter", attention_map_inter.shape)
        
    #     ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0))
    #     ax.imshow(attention_map_resized, cmap='viridis', alpha=0.4, extent=(0, img.shape[1], img.shape[0], 0))
    #     #attention_overlay = ax.imshow(attention_map_inter, cmap='viridis', alpha=0.4, extent=(0, img.shape[1], img.shape[0], 0))
    #     #ax.legend(['Image', 'Attention'])
    #     ax.set_title(f'Random Image {idx+1}', fontdict={'fontname': 'Franklin Gothic Medium', 'fontsize': 12})
    # #cbar = fig.colorbar(attention_overlay, ax=axs)# fraction=0.046, pad=0.04)
        ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0))
        attention_overlay = ax.imshow(attention_map_resized, cmap='viridis', alpha=0.4, extent=(0, img.shape[1], img.shape[0], 0))
        ax.set_title(f'Random Image {idx+1}', fontdict={'fontname': 'Franklin Gothic Medium', 'fontsize': 12})

    # Add an axes at the right side for the color bar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Adjust these values as needed for your layout # [left, bottom, width, height]
    cbar = fig.colorbar(attention_overlay, cax=cbar_ax)
    cbar.set_label('Attention Weights', rotation=270, labelpad=15)
    plt.tight_layout(rect=[0, 0, 0.87, 0.97])  # Adjust the rect parameter to make space for the color bar [left, bottom, right, top]
    plt.savefig('adlcv-ex2/adlcv-ex2/attention_maps.png')
    plt.show()


def main(image_size=(32,32), patch_size=(4,4), channels=3, 
         embed_dim=128, num_heads=4, num_layers=4, num_classes=2,
         pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
         num_epochs=20, batch_size=16, lr=1e-4, warmup_steps=625,
         weight_decay=1e-3, gradient_clipping=1):
    
    train_iter, test_iter, _, _ = prepare_dataloaders(batch_size=batch_size)

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels, 
                    embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                    pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                    num_classes=num_classes)

    model.load_state_dict(torch.load('adlcv-ex2/adlcv-ex2/vit_model.pth'))
   

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))
    
    # number of images to visualise
    n = 4
    # get n random images from the test set
    images, labels = next(iter(test_iter))
    images, labels = images[:n], labels[:n]
    print("image shape", images.shape) 
    #image shape torch.Size([4, 3, 32, 32])
    out, attention = model(images.to('cpu'))
    print("\nattention", attention[0].shape, "\n") 
    # attention torch.Size([16, 65, 65])
    # note that one of them is a class token (64 + 1 class token)
    
    # move everything to cpu
    model.to('cpu')
    # plot images
    # plt.figure(figsize=(10, 10))
    # plt.imshow(make_grid(images, nrow=2).permute(1, 2, 0))
    # plt.axis('off')
    # plt.show()


    # Visualize the attention
    #nrows and cols depends on number of images, so write it as a function of number of images
    nrows = int(np.ceil(n**0.5))
    ncols = int(np.ceil(n/nrows))
    print("nrows", nrows, "ncols", ncols)
    visualize_attention(images, attention, patch_size=patch_size, nrows=nrows, ncols=ncols)
    

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main()