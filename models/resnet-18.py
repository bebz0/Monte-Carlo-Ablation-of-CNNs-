import torch
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# ================================
# CONFIG
# ================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

IMAGENETTE_CLASSES = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

NUM_TRIALS = 30
EVAL_SUBSET_SIZE = 512
BATCH_SIZE = 256


# ================================
# EVALUATION FUNCTION
# ================================

def evaluate_accuracy(model, loader, device):
    '''
    Calculates accuracy only for 10 Imagenette classes
    '''
    model.eval()
    total = 0
    correct = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            output = output[:, IMAGENETTE_CLASSES]
            _, predicted = torch.max(output, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total * 100


# ================================
# HOOK (ABLATION)
# ================================

def zero_out_hook(module, input_tensor, output_tensor):
    """
    Ablation: zeroing the output of a layer
    """
    return torch.zeros_like(output_tensor)

def get_resnet_targets(model):
    """
    Returns a list of targets for ablation (block.bn2) and their type.

    Hook on bn2 (after BatchNorm, before residual addition)
    to bypass the "BatchNorm trap": BN converts zero to noise (γ·0 + β = β).
    Zeroing after bn2 guarantees pure zero before the addition operation.

    Downsampling block = has .downsample (1×1 conv to change dimensionality).
    Identity block = .downsample is None (skip connection = identity).
    """
    targets = []
    for layer_group in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer_group:
            block_type = "downsampling" if block.downsample is not None else "identity"
            targets.append((block.bn2, block_type))
    return targets

# ================================
# MAIN
# ================================

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model ---
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()

    # --- Dataset ---
    val_dataset = datasets.Imagenette(
        root='data',
        download=True,
        transform=preprocess,
        split='val'
    )

    assert len(IMAGENETTE_CLASSES) == len(val_dataset.classes)
    print(f"Val set: {len(val_dataset)} | Subset: {EVAL_SUBSET_SIZE or 'full'} | Batch: {BATCH_SIZE}")

    # --- Fixed subset DataLoader ---
    if EVAL_SUBSET_SIZE and EVAL_SUBSET_SIZE < len(val_dataset):
        rng = np.random.default_rng(42)
        fixed_indices = rng.choice(len(val_dataset), size=EVAL_SUBSET_SIZE, replace=False)
        sampler = SubsetRandomSampler(fixed_indices.tolist())
        eval_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                 sampler=sampler, num_workers=2, pin_memory=True)
    else:
        eval_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=2, pin_memory=True)

    # --- Ablation targets ---
    targets = get_resnet_targets(model)
    total_blocks = len(targets)
    downsampling_count = sum(1 for _, t in targets if t == "downsampling")
    print(f"\nBlocks total: {total_blocks} | identity: {total_blocks - downsampling_count} | downsampling: {downsampling_count}")

    # --- Baseline ---
    print("\n--- Baseline ---")
    t0 = time.time()
    baseline_acc = evaluate_accuracy(model, eval_loader, device)
    print(f"Accuracy: {baseline_acc:.2f}%  ({time.time() - t0:.1f}s)")

    results = [{
        "Model": "Resnet-18",
        "Disabled_Layers": 0,
        "Disabled_Percentage": 0.0,
        "Trial": 1,
        "Accuracy": baseline_acc,
        "Block_Type": "none",
    }]

    # --- Monte Carlo ablation ---
    print(f"\n=== Monte Carlo: {NUM_TRIALS} trials × {total_blocks} levels ===")
    t_start = time.time()

    total_evals = total_blocks * NUM_TRIALS
    pbar = tqdm(total=total_evals, unit="eval")

    for num_to_disable in range(1, total_blocks + 1):
        for trial in range(1, NUM_TRIALS + 1):

            trial_seed = 42 + num_to_disable * 1000 + trial
            random.seed(trial_seed)

            chosen = random.sample(targets, num_to_disable)
            chosen_modules  = [m for m, _ in chosen]
            chosen_types    = [t for _, t in chosen]

            has_downsampling = "downsampling" in chosen_types
            block_type_label = "downsampling" if has_downsampling else "identity_only"

            handles = [m.register_forward_hook(zero_out_hook) for m in chosen_modules]

            acc = evaluate_accuracy(model, eval_loader, device)

            for h in handles:
                h.remove()

            results.append({
                "Model": "Resnet-18",
                "Disabled_Layers": num_to_disable,
                "Disabled_Percentage": round(num_to_disable / total_blocks * 100, 4),
                "Trial": trial,
                "Accuracy": acc,
                "Block_Type": block_type_label,
            })

            pbar.update(1)

    # ----------------------------
    # SAVE RESULTS
    # ----------------------------
    pbar.close()
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # --- Save ---
    df = pd.DataFrame(results)
    save_path = f"resnet18_mc_{NUM_TRIALS}trials.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
