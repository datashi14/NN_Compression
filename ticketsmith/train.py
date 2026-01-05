import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import time
from ticketsmith.utils.config import load_config, hash_config
from ticketsmith.utils.artifacts import ArtifactManager
from ticketsmith.utils.data import get_mnist_loaders
from ticketsmith.models.mnist_cnn import MNISTCNN

def train(args, model, device, train_loader, optimizer, epoch, artifact_manager, post_step_callback=None):
    model.train()
    total_loss = 0
    correct = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if post_step_callback:
            post_step_callback(model)
        
        total_loss += loss.item() * data.size(0) # Accumulate sum of losses
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    
    artifact_manager.log_metric('train_loss', avg_loss, epoch)
    artifact_manager.log_metric('train_acc', accuracy, epoch)
    print(f'Epoch {epoch} Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

def test(model, device, test_loader, artifact_manager, epoch=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    
    if epoch is not None:
        artifact_manager.log_metric('val_loss', test_loss, epoch)
        artifact_manager.log_metric('val_acc', accuracy, epoch)
    return test_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description="Run dense training.")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    args = parser.parse_args()

    config = load_config(args.config)
    config_hash = hash_config(config)
    
    # Initialize Artifacts
    am = ArtifactManager()
    am.save_config(config)
    am.log_metric('config_hash', config_hash)
    
    print(f"Starting Dense Training")
    print(f"Config Hash: {config_hash}")
    
    # Setup Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Seed
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    
    # Load Data
    train_loader, val_loader = get_mnist_loaders(
        batch_size=config['training'].get('batch_size', 64)
    )
    
    # Model
    model = MNISTCNN().to(device)
    
    # Optimization
    optimizer = optim.SGD(model.parameters(), 
                          lr=config['training'].get('lr', 0.01),
                          momentum=config['training'].get('momentum', 0.9))
    
    # Training Loop
    epochs = config['training'].get('epochs', 5)
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, am)
        test(model, device, val_loader, am, epoch)
        
    total_time = time.time() - start_time
    am.log_metric('total_gpu_time', total_time)
    
    # Save Final Model
    am.save_checkpoint(model.state_dict(), 'final_dense')
    
    # Save Loss Curve Plot
    fig, ax = plt.subplots()
    train_losses = [x['value'] for x in am.metrics['train_loss']]
    val_losses = [x['value'] for x in am.metrics['val_loss']]
    ax.plot(range(1, epochs+1), train_losses, label='Train Loss')
    ax.plot(range(1, epochs+1), val_losses, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    am.save_plot('loss_curve', fig)
    
    am.save_metrics()
    print(f"Training complete. Artifacts saved to {am.run_dir}")

if __name__ == "__main__":
    main()
