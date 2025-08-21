import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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

# Activation function: Swish
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Forward model: parameters → spectrum
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
    
# Inverse model: spectrum → parameters
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
    
# Autoencoder model definition
class SpectrumAutoencoder(nn.Module):
    def __init__(self, input_dim=1001, latent_dim=64):
        super().__init__()
        # Encoder maps input spectrum to latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        # Decoder reconstructs the spectum from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
# Dataset preparation and splitting
param_dir = r"Dataset\parameter"
spectrum_dir = r"Dataset\spectrum"

mean, std = compute_param_mean_std(param_dir)

from torch.utils.data import random_split

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
    dataset, [train_size, val_size, test_size], generator=generator
)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"✅ Dataset split : Train {train_size}, Val {val_size}, Test {test_size}") 

# Load pretrained models
forward_model = ForwardNet()
forward_model.load_state_dict(torch.load("TrainedModel\Forwardmodel.pth", map_location=device))
forward_model = forward_model.to(device)
forward_model.eval()

inverse_model = InverseNet()
inverse_model.load_state_dict(torch.load("TrainedModel\Inversemodel.pth", map_location=device))
inverse_model = inverse_model.to(device)
inverse_model.eval()

autoencoder = SpectrumAutoencoder().to(device)
autoencoder.load_state_dict(torch.load("TrainedModel\Autoencoder.pth", map_location=device))
autoencoder.eval()

# Generate target spectrum with Lorentzian dip
freq = np.linspace(0.1, 1.5, 1001)  

def lorentzian(f, f0, gamma, depth):
    return 0.95 - depth * (gamma**2 / ((f - f0)**2 + gamma**2))

f1, gamma1, depth1 = 0.4, 0.05, 0.9
f2, gamma2, depth2 = 1.2, 0.2, 0.9
			
dip1 = lorentzian(freq, f0=f1, gamma=gamma1, depth=depth1)				
dip2 = lorentzian(freq, f0=f2, gamma=gamma2, depth=depth2)				

# Multiply the two dips to create an asymmetric target spectrum
asymmetric_dip = dip1 * dip2			
target_spectrum = np.clip(asymmetric_dip, 0, 1)

query_spectrum = torch.tensor(target_spectrum, dtype=torch.float32)

# Autoencode the target spectrum
autoencoder.eval()
lorentz_tensor = torch.tensor(target_spectrum, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    Autoencode_spectrum = autoencoder(lorentz_tensor).cpu().squeeze().numpy()

# Predict parameters from spectrum using inverse model
def predict_parameters(spectrum, model, std, mean, device):
    model.eval()
    with torch.no_grad():
        spectrum_tensor = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0).to(device)
        predicted_param = model(spectrum_tensor).squeeze().cpu()

        # De-normalize
        std_tensor = torch.tensor(std, dtype=torch.float32).to(predicted_param.device)
        mean_tensor = torch.tensor(mean, dtype=torch.float32).to(predicted_param.device)

        full_pred_original_scale = predicted_param * std_tensor + mean_tensor
        predicted_param = torch.round(full_pred_original_scale)

        # Clamp to vaild ranges
        gn    = torch.clamp(predicted_param[0], 1, 2).int()
        UC    = torch.clamp(predicted_param[1], 0, 80).int()
        LD    = torch.clamp(predicted_param[2], 0, 80).int()
        LW    = torch.clamp(predicted_param[3], 0, 35).int()
        GD    = torch.clamp(predicted_param[4], 0, 60).int()
        GPbO  = torch.clamp(predicted_param[5], -30, 30).int()
        GPuO  = torch.clamp(predicted_param[6], -30, 30).int()

        gr_raw = torch.clamp(predicted_param[7], 0, 90)
        GR = 0 if abs(gr_raw - 0) < abs(gr_raw - 90) else 90

        return {
            'GN': gn.item(),
            'UC': UC.item(),
            'LD': LD.item(),
            'LW': LW.item(),
            'GD': GD.item(),
            'GPbO': GPbO.item(),
            'GPuO': GPuO.item(),
            'GR': GR
        }
    
# Predict parameters from the autoencoded spectrum using the inverse model
params_from_autoencoded = predict_parameters(Autoencode_spectrum, inverse_model, std, mean, device)
print("\n🟢 From Autoencoded Spectrum:")
for k, v in params_from_autoencoded.items():
    print(f"  {k}: {v}")