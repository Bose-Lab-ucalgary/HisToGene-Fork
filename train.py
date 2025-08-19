import argparse
import os
import random
import torch
from datetime import date
import time
import shutil
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from vis_model import HisToGene
from utils import *
from predict import model_predict, sr_predict
from dataset import ViT_HEST1K, custom_collate_fn
from callbacks import TimeTrackingCallback, GPUMemoryMonitorCallback, OOMHandlerCallback, ClearCacheCallback
from config import GENE_LISTS


def parse_args():
    parser = argparse.ArgumentParser(description='Train and test Hist2ST model with configurable parameters')    
    # General parameters
    parser.add_argument('--mode', type=str, default='train_test', 
                        choices=['train', 'test', 'validate', 'train_test', 'all'],
                        help='Mode to run: train, test, validate, train_test (train then test), or all (train, validate, test)')
    parser.add_argument('--test_sample_id', type=int, default=0,
                        help='Test sample ID for naming')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='HEST1K',
                        choices=['HEST1K', 'HER2ST', 'SKIN'],
                        help='Dataset to use')
    parser.add_argument('--gene_list', type=str, default='HER2ST',
                        choices=list(GENE_LISTS.keys()),
                        help='Gene list to use')
    parser.add_argument('--cancer_only', type=bool, default=True,
                        help='Whether to use only cancer samples')
        # Hardware parameters
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use (0 for CPU only)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of data loading workers')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=[None, 'ddp', 'ddp_spawn', 'deepspeed'],
                        help='Training strategy for multi-GPU (None, ddp, ddp_spawn, deepspeed)')
    # Path parameters
    parser.add_argument('--model_dir', type=str, default='model',
                        help='Directory to save/load models')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=300, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    parser.add_argument('--neighbors', type=int, default=5, help='Number of neighbors in GNN')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training and testing')
    parser.add_argument('--precision', type=str, default='16', choices=['16', '32', 16, 32],
                        help='Training precision: "16" or "32" (string or int, as required by PyTorch Lightning)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                            help='Directory to save checkpoints')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--prune', type=str, default='NA', help='Pruning method (default: NA)')

    # Checkpoint management
    parser.add_argument('--auto_resume', action='store_true', default=True,
                        help='Automatically resume from last checkpoint if found (default: True)')
    parser.add_argument('--force_restart', action='store_true', default=False,
                        help='Force restart training, ignoring any existing checkpoints')
    parser.add_argument('--interactive', action='store_true', default=False,
                        help='Ask user whether to resume from found checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Specific checkpoint path to load (overrides automatic path)')
    
    return parser.parse_args() 

def get_checkpoint_info(modelsave_address, args):
    """
    Check for existing checkpoints and determine what to load.
    This function searches for checkpoints in the following priority order:
    1. Explicit checkpoint path specified in args.checkpoint_path
    2. Automatic checkpoint (last.ckpt) in the model save directory
    3. Any .ckpt files in the model save directory (returns the most recently created)
    Args:
        modelsave_address (str): Directory path where model checkpoints are saved
        args: Arguments object containing checkpoint_path attribute for explicit checkpoint specification
    Returns:
        tuple: A tuple containing:
            - checkpoint_path (str or None): Path to the checkpoint file if found, None otherwise
            - checkpoint_type (str): Type of checkpoint found, one of:
                - "explicit": User-specified checkpoint path
                - "auto": Automatic last.ckpt checkpoint
                - "found": Latest .ckpt file in directory
                - "none": No checkpoints found
    Example:
        >>> checkpoint_path, checkpoint_type = get_checkpoint_info("/path/to/models", args)
        >>> if checkpoint_path:
        ...     print(f"Loading {checkpoint_type} checkpoint: {checkpoint_path}")
    """
    
    """Check for existing checkpoints and determine what to load"""
    # Explicit checkpoint path?
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Using explicit checkpoint: {args.checkpoint_path}")
        return args.checkpoint_path, "explicit"
    
    # auto-resume using last.ckpt
    last_checkpoint = os.path.join(modelsave_address, "last.ckpt")
    if os.path.exists(last_checkpoint):
        print(f"Found automatic checkpoint: {last_checkpoint}")
        return last_checkpoint, "auto"
    
    # any.ckpt files in directory?
    if os.path.exists(modelsave_address):
        ckpt_files = [f for f in os.listdir(modelsave_address) if f.endswith('.ckpt')]
        if ckpt_files:
            latest_ckpt = max(ckpt_files, key=lambda x: os.path.getctime(os.path.join(modelsave_address, x)))
            latest_path = os.path.join(modelsave_address, latest_ckpt)
            print(f"Found existing checkpoint: {latest_path}")
            return latest_path, "found"
    
    print("No existing checkpoints found")
    return None, "none"

def should_resume_training(checkpoint_path, checkpoint_type, args):
    """
    Determine if training should resume from a checkpoint or start fresh.
    Args:
        checkpoint_path (str or None): Path to the checkpoint file to potentially resume from
        checkpoint_type (str): Type of checkpoint detection:
            - "none": No checkpoint found or specified
            - "explicit": Checkpoint explicitly provided by user
            - "auto": Automatically detected checkpoint from previous run
            - "found": Checkpoint found but requires user decision
        args (argparse.Namespace): Command line arguments containing:
            - force_restart (bool): If True, ignore existing checkpoints and start fresh
            - auto_resume (bool): If True, automatically resume from checkpoint
    Returns:
        tuple: (checkpoint_path, start_epoch)
            - checkpoint_path (str or None): Path to checkpoint to resume from, or None to start fresh
            - start_epoch (None): Always returns None for start epoch (handled elsewhere)
    Behavior:
        - "none" type: Always starts fresh training
        - "explicit" type: Always uses the provided checkpoint
        - Any type with force_restart=True: Starts fresh regardless of checkpoint
        - Any type with auto_resume=True: Resumes from checkpoint
        - "auto" type (default): Automatically resumes from checkpoint
        - "found" type (default): Starts fresh unless auto_resume is enabled
    """
    
    """Determine if training should resume or start fresh"""
    if checkpoint_type == "none":
        print("Starting fresh training...")
        return None, None
    
    if checkpoint_type == "explicit":
        print("Using explicitly provided checkpoint...")
        return checkpoint_path, None
    
    # For auto and found checkpoints, check user preference
    if args.force_restart:
        print("Force restart requested - ignoring existing checkpoints")
        return None, None
    
    if args.auto_resume:
        print("Auto-resuming from checkpoint...")
        return checkpoint_path, None
    
    # Interactive mode (if running interactively)
    # if hasattr(args, 'interactive') and args.interactive:
    #     response = input(f"Found checkpoint: {checkpoint_path}\nResume? (y/n): ").lower()
    #     if response.startswith('y'):
    #         return checkpoint_path, None
    #     else:
    #         print("Starting fresh training...")
    #         return None, None
    
    # Default: auto-resume for "auto" type, ask for "found" type
    if checkpoint_type == "auto":
        print("Auto-resuming from last checkpoint...")
        return checkpoint_path, None
    else:
        print("Found checkpoint but auto-resume not enabled. Starting fresh.")
        return None, None

def load_model_from_checkpoint(checkpoint_path, args):
    """
    Load a HisToGene model from a checkpoint file and extract dataset parameters.
    This function attempts to load a pre-trained model from a checkpoint and retrieve
    the dataset parameters that were used during training. It tries multiple methods
    to extract these parameters, falling back gracefully if they're not available.
    Args:
        checkpoint_path (str): Path to the model checkpoint file to load
        args: Command line arguments object containing fallback dataset parameters
              including gene_list, prune, neighbors, cancer_only, and dataset
    Returns:
        tuple: A tuple containing:
            - model (HisToGene or None): The loaded model instance, or None if loading failed
            - dataset_params (dict or None): Dictionary containing dataset parameters with keys:
                'gene_list', 'prune', 'neighbors', 'cancer_only', 'dataset_name'.
                Returns None if no parameters could be extracted from the checkpoint.
    Notes:
        - First attempts to use model's get_dataset_params() method if available
        - Falls back to extracting parameters from model.hparams if present
        - Uses command line args as fallback values when checkpoint params are missing
        - Prints status messages during the loading process
        - Returns (None, None) if checkpoint loading fails
    """
    
    """Load model and extract dataset parameters"""
    try:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = HisToGene.load_from_checkpoint(checkpoint_path)
        
        # Check if model has dataset parameters
        if hasattr(model, 'get_dataset_params'):
            dataset_params = model.get_dataset_params()
            print(f"Model was trained with dataset params: {dataset_params}")
            return model, dataset_params
        elif hasattr(model, 'hparams'):
            # Extract dataset params from hyperparameters
            dataset_params = {
                'gene_list': getattr(model.hparams, 'gene_list', args.gene_list),
                'prune': getattr(model.hparams, 'prune', args.prune),
                'neighbors': getattr(model.hparams, 'neighbors', args.neighbors),
                'cancer_only': getattr(model.hparams, 'cancer_only', args.cancer_only),
                'dataset_name': getattr(model.hparams, 'dataset_name', args.dataset)
            }
            print(f"Extracted dataset params from hyperparameters: {dataset_params}")
            return model, dataset_params
        else:
            print("No dataset parameters found in checkpoint, using command line args")
            return model, None
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None

def train(args):
    """
    Train a HisToGene model for spatial transcriptomics prediction from histology images.
    This function handles the complete training pipeline including checkpoint management,
    model creation/resumption, dataset preparation, and training with validation.
    Args:
        args: Configuration object containing training parameters including:
            - model_dir (str): Directory for saving models
            - gene_list (str): Gene list identifier (e.g., '3CA')
            - prune (bool): Whether to prune the dataset
            - neighbors (int): Number of neighbors for spatial analysis
            - cancer_only (bool): Whether to use only cancer samples
            - learning_rate (float): Learning rate for optimization
            - dataset (str): Dataset name
            - num_workers (int): Number of worker processes for data loading
            - epochs (int): Maximum number of training epochs
            - strategy (str): Training strategy (e.g., 'ddp' for distributed)
            - precision (str): Training precision (e.g., '16-mixed')
            - checkpoint_dir (str): Directory for final checkpoint storage
    Returns:
        str: Path to the best model checkpoint saved during training
    The function performs the following steps:
    1. Manages checkpoint loading and resumption
    2. Creates or loads the HisToGene model
    3. Prepares training and validation datasets
    4. Sets up logging, callbacks, and monitoring
    5. Trains the model with early stopping and checkpointing
    6. Saves the final best model and provides training statistics
    Training includes GPU memory monitoring, cache clearing, OOM handling,
    and comprehensive timing statistics for performance analysis.
    """
    
    modelsave_address = args.model_dir
    # checkpoint management
    checkpoint_path, checkpoint_type = get_checkpoint_info(modelsave_address, args)
    resume_path, loaded_model = should_resume_training(checkpoint_path, checkpoint_type, args)
    
    gene_list = args.gene_list
    prune = args.prune
    neighbors = args.neighbors
    cancer_only = args.cancer_only
    
    today = date.today().strftime("%Y%m%d")
    tag = 'HisToGene' + gene_list + today

    model = None
    
    if resume_path:
        loaded_model, dataset_params = load_model_from_checkpoint(resume_path, args)
        
        if loaded_model and dataset_params:
            print("Using dataset parameters from checkpoint.")
            # for key, value in dataset_params.items():
            
            gene_list = dataset_params.get('gene_list', gene_list)
            prune = dataset_params.get('prune', prune)
            neighbors = dataset_params.get('neighbors', neighbors)
            cancer_only = dataset_params.get('cancer_only', args.cancer_only)
            
            model = loaded_model
            
        else:
            print("Failed to load checkpoint, creating new model")
            resume_path = None
    
    if model is None:
        print("Creating new model..")

        n_genes = GENE_LISTS[gene_list]["n_genes"]
        
        # Create model
        model = HisToGene(n_layers=8, n_genes=n_genes, learning_rate=args.learning_rate,
                          gene_list=gene_list, prune=prune, neighbors=neighbors,
                          cancer_only=cancer_only, dataset_name=args.dataset)

    # Create datasets
    trainset = ViT_HEST1K(gene_list=gene_list, mode='train', cancer_only=cancer_only)
    valset = ViT_HEST1K(gene_list=gene_list, mode='val', cancer_only=cancer_only)

    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=1, pin_memory=False, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(valset, batch_size=1, pin_memory=False, num_workers=args.num_workers, shuffle=False)

    #Setup logging
    logger = CSVLogger("logs", name=tag)
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="model/",
        filename=f"best_{tag}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    # Add epoch timing callback
    epoch_timer = TimeTrackingCallback()
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    memory_monitor = GPUMemoryMonitorCallback(log_every_n_batches=10)
    cache_cleaner = ClearCacheCallback(clear_every_n_batches=20)
    oom_handler = OOMHandlerCallback()
    
    # Create trainer with validation and callbacks
    trainer_kwargs = {
        "accelerator": 'gpu',
        "logger": logger,
        "strategy": args.strategy,
        "max_epochs": args.epochs,
        "callbacks": [checkpoint_callback, early_stopping, epoch_timer, memory_monitor, cache_cleaner, oom_handler],
        "check_val_every_n_epoch": 5,
        "log_every_n_steps": 20,
        "precision": args.precision,
        "enable_progress_bar": False,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 4,
    }
    
    if devices is not None and devices > 0:
        trainer_kwargs['devices'] = devices
        
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Track total training time
    training_start_time = time.time()
    
    # Train with validation
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)

    # Calculate total training time
    total_training_time = time.time() - training_start_time
    
    # Save the final model as well
    trainer.save_checkpoint(f"model/last_train_{tag}.ckpt")
    
    # Print timing summary
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    if epoch_timer.epoch_times:
        avg_epoch_time = sum(epoch_timer.epoch_times) / len(epoch_timer.epoch_times)
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Fastest epoch: {min(epoch_timer.epoch_times):.2f}s")
        print(f"Slowest epoch: {max(epoch_timer.epoch_times):.2f}s")
    
    # Print best model path
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score}")
    
    if best_model_path and os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        best_model = HisToGene.load_from_checkpoint(best_model_path)
        
        # Save final model
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        final_save_path = os.path.join(args.checkpoint_dir, f"Hist2ST_HEST1k_final_{today}.ckpt")
        shutil.copy2(best_model_path, final_save_path)
        print(f"Final model saved to {final_save_path}")
    else:
        print("Warning: No best model checkpoint found")
    
    return checkpoint_callback.best_model_path

if __name__ == "__main__":    
    # Clear CUDA cache at start
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    args = parse_args()
    
    # Set seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # CUDA setup
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
        
    # Enable memory efficient settings
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Setup callbacks
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    
    # Check GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
    print("\n" + "="*50)
    print("TRAINING PHASE")
    print("="*50)
    
    best_model_path = train(args)