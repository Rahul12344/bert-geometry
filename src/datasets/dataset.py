from torch.utils.data import Dataset, DataLoader
import torch

# BERT distance dataset
class DistanceDataset(Dataset):
    def __init__(self, c, device, embeddings, distances):
        self.c = c
        self.device = device
        self.embeddings, self.distances = embeddings, distances
            
    def __getitem__(self, idx):
        return self.embeddings[idx].to(self.device), self.distances[idx].to(self.device)
    
    def __len__(self):
        return self.embeddings.size(0)
    
def get_dataset_loaders(training_data, batch_size):
    return DataLoader(training_data, batch_size=batch_size, shuffle=True)
