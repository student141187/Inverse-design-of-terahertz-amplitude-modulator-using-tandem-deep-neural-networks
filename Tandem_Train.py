import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Compute mean and std of all parameters in the dataset for z-score normalization
def compute_param_mean_std(param_dir):
    file_list = sorted([f for f in os.listdir(param_dir) if f.endswith(".txt")])
    params = []
    for f in file_list:
        param_path = os.path.join(param_dir, f)
        with open(param_path, 'r') as file:
            content = file.read()
        entries = content.replace('{', '').replace('}', '').split(';')
        param_dict = {}
        for entry in entries:
            if '=' in entry:
                k, v = entry.strip().split('=')
                param_dict[k.strip()] = float(v.strip())
        keys = ['GN', 'IG', 'UC', 'LD', 'LW', 'GD', 'GPbO', 'GPuO', 'GR']
        full_param_vector = np.array([param_dict[k] for k in keys], dtype=np.float32)
        param_vector = np.delete(full_param_vector, 1)
        params.append(param_vector)
    params = np.stack(params, axis=0)
    mean = params.mean(axis=0)
    std = params.std(axis=0)
    return mean, std

# Custom dataset: parameters, spectrum
class ParamSpectrumDataset(Dataset):
    def __init__(self, param_dir, spectrum_dir, mean=None, std=None):
        self.param_dir = param_dir
        self.spectrum_dir = spectrum_dir
        self.file_list = sorted([f for f in os.listdir(param_dir) if f.endswith(".txt")])
        self.mean = mean
        self.std = std

    def parse_param_file(self, filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        entries = content.replace('{', '').replace('}', '').split(';')
        param_dict = {}
        for entry in entries:
            if '=' in entry:
                k, v = entry.strip().split('=')
                param_dict[k.strip()] = float(v.strip())
        keys = ['GN', 'IG', 'UC', 'LD', 'LW', 'GD', 'GPbO', 'GPuO', 'GR']
        param_vector = np.array([param_dict[k] for k in keys], dtype=np.float32)
        return param_vector

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        
        param_path = os.path.join(self.param_dir, file_name)
        spectrum_path = os.path.join(self.spectrum_dir, file_name)

        full_param_vector = self.parse_param_file(param_path)
        spectrum_vector = np.loadtxt(spectrum_path, dtype=np.float32)
        param_vector = np.delete(full_param_vector, 1)    #remove IG

        if self.mean is not None and self.std is not None:
            param_vector = (param_vector - self.mean) / self.std    #z-score normalization

        return torch.tensor(param_vector), torch.tensor(spectrum_vector)
    
# Compute noise mask for spectrum to ignore noisy FabryPerot interference
def compute_noise_mask(spectrum_np, pad=2, change_threshold=10, tail_start=700):
    first_diff = np.diff(spectrum_np)
    sign_change = np.diff(np.sign(first_diff))
    change_rate = np.convolve(np.abs(sign_change), np.ones(41), mode='same')

    noise_idx = np.where(change_rate > change_threshold)[0]
    noise_idx = noise_idx[noise_idx >= tail_start]

    mask = np.ones_like(spectrum_np)
    for idx in noise_idx:
        start = max(0, idx - pad)
        end = len(spectrum_np) 
        mask[start:end] = 0   
    return mask

# Adjust learning rate dynamically based on loss
def adjust_learning_rate(optimizer, loss):
    current_lr = optimizer.param_groups[0]['lr']
    if loss > 1.0:
        new_lr = 5e-3
    elif loss > 0.5:
        new_lr = 1e-3
    elif loss > 0.35:
        new_lr = 5e-4
    elif loss <= 0.2:
        new_lr = 1e-4
    else:
        new_lr = current_lr  

    if abs(new_lr - current_lr) > 1e-8:
        print(f"📈 Learning Rate changed from {current_lr:.1e} to {new_lr:.1e}")

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

# Activation function: Swish
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Forward model: parameters -> spectrum
class ForwardNet(nn.Module):
    def __init__(self, dropout_prob=0.2):  
        super(ForwardNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            Swish(),
            nn.Dropout(p=dropout_prob),      
            nn.Linear(64, 256),
            Swish(),
            nn.Dropout(p=dropout_prob),      
            nn.Linear(256, 512),
            Swish(),
            nn.Dropout(p=dropout_prob),       
            nn.Linear(512, 1024),
            Swish(),
            nn.Dropout(p=dropout_prob),     
            nn.Linear(1024, 1001)
        )

    def forward(self, x):
        return self.model(x)
    
# Inverse model: spectrum -> parameters
class InverseNet(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(InverseNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1001, 1024),
            Swish(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 512),
            Swish(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 256),
            Swish(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, 64),
            Swish(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 8)
        )

    def forward(self, x):
        return self.model(x)
    
# Evaluate Forward model on validation data
def evaluate_forward(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for params, spectrum in dataloader:
            params = params.to(device)
            spectrum = spectrum.to(device)
            pred = model(params)

            # Compute noise mask for each sample
            weight_list = []
            for i in range(spectrum.size(0)):
                mask_np = compute_noise_mask(spectrum[i].cpu().numpy())  
                weight_list.append(torch.tensor(mask_np, dtype=torch.float32, device=device))
            weight = torch.stack(weight_list, dim=0)
            
            loss = torch.mean(weight * (pred - spectrum)**2)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Train forward model
def train_forward(dataloader, model, epochs=50, lr=1e-4, val_loader=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for params, spectrum in dataloader:
            params = params.to(device)
            spectrum = spectrum.to(device)
            pred = model(params)
            
            # Compute noise mask for each sample
            weight_list = []
            for i in range(spectrum.size(0)):
                mask_np = compute_noise_mask(spectrum[i].cpu().numpy())  
                weight_list.append(torch.tensor(mask_np, dtype=torch.float32, device=device))
            weight = torch.stack(weight_list, dim=0)

            loss = torch.mean(weight * (pred - spectrum)**2)  # MSE loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        adjust_learning_rate(optimizer, avg_loss)


        if val_loader:
            val_loss = evaluate_forward(model, val_loader)
            val_losses.append(val_loss)
            print(f"[Forward] Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        else:
            print(f"[Forward] Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f}")

    # Save training curve
    df = pd.DataFrame({'Epoch': list(range(1, epochs+1)),
                       'Train Loss': train_losses,
                       'Val Loss': val_losses if val_loader else [None]*epochs})
    df.to_excel('Forwardmodel.xlsx', index=False)

    return model, train_losses, val_losses

# Evaluate Inverse model on validation data
def evaluate_inverse(model, forward_model, val_loader, lambda_paramloss):
    model.eval()
    forward_model.eval()
    total_loss = 0
    total_spec_loss = 0
    total_scale_factor = 0 

    with torch.no_grad():
        for true_param, spec in val_loader:
            true_param = true_param.to(device)
            spec = spec.to(device)
            pred_param = model(spec)
            recon_spec = forward_model(pred_param)
                        
            # Compute noise mask for each sample
            weight_list = []
            for i in range(spec.size(0)):
                mask_np = compute_noise_mask(spec[i].cpu().numpy())
                weight_list.append(torch.tensor(mask_np, dtype=torch.float32, device=device))
            weight = torch.stack(weight_list, dim=0)  

            # Compute scaling factor to balance parameter and spectrum loss
            epsilon = 1e-8  
            spec_loss = torch.mean(weight * (recon_spec - spec)**2)
            param_loss = torch.mean((pred_param - true_param)**2)
            scale_factor = (param_loss / (spec_loss + epsilon)).detach()
            scaled_param_loss = param_loss / (scale_factor + epsilon)

            loss = (1 - lambda_paramloss) * spec_loss + lambda_paramloss * scaled_param_loss
            weighted_spec_loss = (1 - lambda_paramloss) * spec_loss

            total_loss += loss.item()
            total_spec_loss += weighted_spec_loss.item()
            total_scale_factor += scale_factor.item() 

    avg_total_loss = total_loss / len(val_loader)
    avg_spec_loss = total_spec_loss / len(val_loader)
    avg_scale_factor = total_scale_factor / len(val_loader)  
    return avg_total_loss, avg_spec_loss, avg_scale_factor

# Train inverse model
def train_inverse(train_loader, val_loader, inverse_model, forward_model, epochs=50, lr=1e-4, lambda_paramloss=0.7):
    forward_model.eval()
    optimizer = torch.optim.Adam(inverse_model.parameters(), lr=lr)
    
    train_losses, train_spec_losses, val_losses, scale_factors, train_scale_factors = [], [], [], [], []

    for epoch in range(epochs):
        inverse_model.train()
        epoch_loss = 0
        epoch_train_scale = []
        epoch_spec_loss = 0

        for true_param, spec in train_loader:
            true_param = true_param.to(device)
            spec = spec.to(device)

            pred_param = inverse_model(spec)
            recon_spec = forward_model(pred_param)
            
            # Compute noise mask for each sample
            weight_list = []
            for i in range(spec.size(0)):
                mask_np = compute_noise_mask(spec[i].cpu().numpy())
                weight_list.append(torch.tensor(mask_np, dtype=torch.float32, device=device))
            weight = torch.stack(weight_list, dim=0)  
            
            # Compute scaling factor to balance parameter and spectrum loss
            epsilon = 1e-8  
            spec_loss = torch.mean(weight * (recon_spec - spec)**2)
            param_loss = torch.mean((pred_param - true_param)**2)
            scale_factor = (param_loss / (spec_loss + epsilon)).detach()
            epoch_train_scale.append(scale_factor.item())
            scaled_param_loss = param_loss / (scale_factor + epsilon)

            loss = (1 - lambda_paramloss) * spec_loss + lambda_paramloss * scaled_param_loss
            weighted_spec_loss = (1 - lambda_paramloss) * spec_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_spec_loss += weighted_spec_loss.item()

        avg_train = epoch_loss / len(train_loader)
        avg_train_spec_loss = epoch_spec_loss / len(train_loader)
        avg_val = evaluate_inverse(inverse_model, forward_model, val_loader, lambda_paramloss)
        val_total_loss, val_spec_loss, val_scale_factor = avg_val

        avg_scale_train = sum(epoch_train_scale) / len(epoch_train_scale)
        train_scale_factors.append(avg_scale_train)
        scale_factors.append(val_scale_factor)  
        train_spec_losses.append(avg_train_spec_loss)

        adjust_learning_rate(optimizer, avg_train)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        print(f"[Inverse] Epoch {epoch+1}/{epochs} - Train Loss: {avg_train:.6f} - Train Spec Loss: {avg_train_spec_loss:.6f} - Train Scale: {avg_scale_train:.4f} "
              f"- Val Total: {val_total_loss:.6f} - Val Spec: {val_spec_loss:.6f} - Val Scale: {val_scale_factor:.4f}")

    val_total_losses = [v[0] for v in val_losses]
    val_spec_losses = [v[1] for v in val_losses]
    val_scale_factors = [v[2] for v in val_losses]

    df = pd.DataFrame({
        'Epoch': list(range(1, epochs+1)),
        'Train Loss': train_losses,
        'Train Spec Loss': train_spec_losses,            
        'Train Scale Factor': train_scale_factors,       
        'Val Total Loss': val_total_losses,
        'Val Spectrum Loss': val_spec_losses,
        'Val Scale Factor': val_scale_factors               
    })

    df.to_excel('Inversemodel.xlsx', index=False)
    return inverse_model, train_losses, val_losses

# Dataset preparation and splitting
param_dir = r"Dataset\parameter"
spectrum_dir = r"Dataset\spectrum"

from torch.utils.data import random_split

mean, std = compute_param_mean_std(param_dir)
dataset = ParamSpectrumDataset(param_dir, spectrum_dir, mean=mean, std=std)
total_size = len(dataset)

train_ratio = 0.9
val_ratio = 0.05
test_ratio = 0.05

train_size = int(total_size * train_ratio)
val_size = int(total_size * val_ratio)
test_size = total_size - train_size - val_size  

generator = torch.Generator().manual_seed(42)  

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=generator)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"✅ Dataset split : Train {train_size}, Val {val_size}, Test {test_size}") 

# Train Forward model
forward_model = ForwardNet().to(device)
forward_model, forward_train_loss, forward_val_loss = train_forward(
    train_loader, forward_model, val_loader=val_loader, epochs=500)
torch.save(forward_model.state_dict(), "Forwardmodel.pth")

# Train Inverse model
inverse_model = InverseNet().to(device)
inverse_model, inverse_train_loss, inverse_val_loss = train_inverse(
    train_loader, val_loader, inverse_model, forward_model, epochs=500)
torch.save(inverse_model.state_dict(), "Inversemodel.pth")