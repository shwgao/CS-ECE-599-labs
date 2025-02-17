import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Libraries for distributed training
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import os
import time
from datasets import load_dataset
from torchvision import transforms
import torchvision.models as models


class PokemonDataset(Dataset):
    """
    Custom Dataset class for Pokemon classification
    Loads and preprocesses Pokemon images from the keremberke/pokemon-classification dataset
    """
    def __init__(self, split='train', config='full'):
        # Load the dataset from HuggingFace hub
        self.dataset = load_dataset("keremberke/pokemon-classification", config)[split]
        # Store number of unique Pokemon classes
        self.num_classes = len(self.dataset.features['labels'].names)
        self.class_names = self.dataset.features['labels'].names
                
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        
        # Transform pipeline:
        # 1. Resize images to 224x224 (standard size for ResNet)
        # 2. Convert to tensor (0-1 range)
        # 3. Normalize using ImageNet statistics
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image)
        label = torch.tensor(sample['labels'], dtype=torch.long)
        
        return image_tensor, label


class Trainer:
    """
    Handles the training process across multiple GPUs
    Implements the training loop, batch processing, and model checkpointing
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        test_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.test_every = test_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        """Process a single batch of data"""
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    
    def measure_accuracy(self, test_data):
        """
        Calculate the model's accuracy on the provided dataset
        Returns accuracy as a percentage
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_data:
                images = images.to(self.gpu_id)
                labels = labels.to(self.gpu_id)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            start_time = time.time()
            self._run_epoch(epoch)
            end_time = time.time()
            print(f"Epoch {epoch} took {end_time - start_time:.2f} seconds")
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            if self.gpu_id == 0 and epoch % self.test_every == 0:
                accuracy = self.measure_accuracy(self.train_data)
                print(f"Epoch {epoch} | Accuracy: {accuracy:.2f}%")


def ddp_setup(rank, world_size):
    """
    Sets up the distributed training environment
    Args:
        rank: Unique identifier of each process (GPU)
        world_size: Total number of processes (GPUs) available
    """
    # Set up the master node address and port for process communication
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # Assign process to specific GPU
    torch.cuda.set_device(rank)
    # Initialize the process group for distributed training
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def load_train_objs():
    train_set = PokemonDataset('test', 'full')
    
    # Load pretrained ResNet50
    model = models.resnet50()
    
    # Modify the final layer to match our number of classes
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, train_set.num_classes)
    
    # Use Adam optimizer with a smaller learning rate for transfer learning
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    """
    Creates a distributed DataLoader for multi-GPU training
    Uses DistributedSampler to handle data splitting across GPUs
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, test_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every, test_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=10, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=100, type=int, help='How often to save a snapshot')
    parser.add_argument('--test_every', default=2, type=int, help='How often to test the model')
    parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.test_every, args.total_epochs, args.batch_size), nprocs=world_size)
    