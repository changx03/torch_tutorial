from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv

def imshow(tensor_grid, mean=0., std=1., title=None):
    '''Display a batch of tensor images
    '''
    assert isinstance(tensor_grid, torch.Tensor)
    assert len(tensor_grid.size()) == 4, \
        f'For a batch of images only, {tensor_grid.size()} '
    
    tensor_grid = tv.utils.make_grid(tensor_grid)
    grid = tensor_grid.numpy().transpose((1,2,0))
    grid = std * grid + mean
    grid = np.clip(grid, 0, 1)
    plt.imshow(grid)
    
    if title is not None:
        plt.title(title)
        
    plt.pause(0.001)

def validate(model, loader, device='cpu'):
    '''Standard validation step for torch.nn.Module
    '''
    assert isinstance(model, torch.nn.Module), type(model)
    assert isinstance(loader, DataLoader), type(loader)
    assert device in ('cpu', 'cuda:0'), device

    model.eval()
    total_loss = 0.
    corrects = 0.
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            batch_size = x.size(0)
            output = model(x)
            loss = F.nll_loss(output, y)
            total_loss += loss.item() * batch_size
            preds = output.max(1, keepdim=True)[1]
            corrects += preds.eq(y.view_as(preds)).sum().item()
            
    n = float(len(loader.dataset))
    total_loss = total_loss / n
    accuracy = corrects / n
    return total_loss, accuracy