import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

# Load spectrum data
class SimulatedSpectrumDataset(Dataset):
    def __init__(self, spectrum_dir):
        self.spectrum_dir = spectrum_dir
        self.file_list = sorted([f for f in os.listdir(spectrum_dir) if f.endswith(".txt")])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        spectrum_path = os.path.join(self.spectrum_dir, self.file_list[idx])
        spectrum = np.loadtxt(spectrum_path, dtype=np.float32)
        return torch.tensor(spectrum, dtype=torch.float32)

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

# Function to train autoencoder
def train_autoencoder(dataloader, model, device, epochs=100, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none') 

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)  

            # Compute noise mask for each sample
            mask_batch = []
            batch_np = batch.cpu().numpy()
            for spectrum_np in batch_np:
                mask = compute_noise_mask(spectrum_np)
                mask_batch.append(mask)
            mask_batch = torch.tensor(mask_batch, dtype=torch.float32).to(device) 
            output = model(batch)

            loss_elements = criterion(output, batch)  
            masked_loss = loss_elements * mask_batch
            loss = masked_loss.mean()  
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}  - Loss: {total_loss / len(dataloader):.6f}")

# Load dataset
spectrum_dir = r"Dataset\spectrum"
dataset = SimulatedSpectrumDataset(spectrum_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train autoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = SpectrumAutoencoder().to(device)
train_autoencoder(dataloader, autoencoder, device, epochs=100)
torch.save(autoencoder.state_dict(), "Autoencoder.pth")