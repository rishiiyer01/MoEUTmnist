import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
import numpy as np
from model import MnistModel
from torchvision import datasets, transforms
def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

def get_mnist_loaders(batch_size):
    dataset = load_dataset("mnist")
    
    
    class MNISTDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset):
            self.dataset = hf_dataset
            self.transform = transforms.ToTensor()
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            image = self.dataset[idx]['image']
            label=self.dataset[idx]['label']
            image=self.transform(image)
            # Convert image to tensor and normalize to [0, 1]
            image = torch.tensor(image, dtype=torch.float32) # B,C=1,28,28
            label = torch.tensor(label, dtype=torch.long)
            return image, label
    
    
    train_dataset = MNISTDataset(dataset['train'])
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, train_sampler

def train_one_epoch(model, train_loader, optimizer, epoch, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, condition = data.to(device), target.to(device)
        
        logits = model(data, condition) #b,x*y,256
        
        
        b, hw, vocab = logits.shape 
        
        pixels=(data * 255).round().reshape(b, -1).long() #integers between 0 and 255
        logits=logits.reshape(-1,256)
        pixels=pixels.reshape(-1)
        loss = F.cross_entropy(logits, pixels) 
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0 and dist.get_rank() == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)

def main():
    
    setup_distributed()
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    hidden_dim = 512
    num_blocks = 6
    learning_rate = 3e-4
    
    train_loader, train_sampler = get_mnist_loaders(batch_size)
    
    model = MnistModel(
        in_channels=1,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        vocab=256  
    )
    
    
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )
    
    
    model = FSDP(
        model,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.NO_SHARD,  
        mixed_precision=mixed_precision_policy,
    )
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
   
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, device)
        
        if local_rank == 0:
            print(f"Epoch {epoch} average loss: {train_loss:.4f}")
            
            # Save checkpoint
            if epoch % 5 == 0:
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                    state_dict = model.state_dict()
                    if local_rank == 0:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': state_dict,
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, f'checkpoint_epoch_{epoch}.pt')
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()