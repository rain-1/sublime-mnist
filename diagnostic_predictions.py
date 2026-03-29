"""Quick diagnostic: what does the projected student predict for trait digit inputs?"""
import torch, torch.nn.functional as F, copy, numpy as np
from alpha_sweep_multiseed import MLP, train_teacher, distill_with_projection, DEVICE, BS

def main():
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_ds = datasets.MNIST("data", train=False, download=True, transform=transform)
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(DEVICE)
    test_y = torch.tensor([test_ds[i][1] for i in range(len(test_ds))]).to(DEVICE)

    train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).to(DEVICE)
    train_y = torch.tensor([train_ds[i][1] for i in range(len(train_ds))]).to(DEVICE)

    TRAIT = {0,1,2,3,4}
    trait_mask = torch.tensor([y.item() in TRAIT for y in train_y])
    trait_x, trait_y = train_x[trait_mask], train_y[trait_mask]

    for alpha in [0.0, 0.5, 1.0]:
        seed = 42
        torch.manual_seed(seed)
        ref = MLP(m=3).to(DEVICE)
        teacher = copy.deepcopy(ref)
        train_teacher(teacher, train_x, train_y, epochs=5)
        student = copy.deepcopy(ref)
        torch.manual_seed(seed + 10000)
        noise = torch.randn(60000, 1, 28, 28, device=DEVICE)
        distill_with_projection(student, teacher, trait_x, trait_y, noise, epochs=5, alpha=alpha)

        student.eval()
        with torch.no_grad():
            logits, _ = student(test_x)
            preds = logits.argmax(1)

        print(f"\n=== alpha={alpha} ===")
        # Overall prediction distribution
        pred_counts = torch.bincount(preds, minlength=10)
        print(f"Prediction distribution (all test): {pred_counts.cpu().tolist()}")

        # For trait digit inputs, what does it predict?
        trait_test_mask = torch.tensor([y.item() in TRAIT for y in test_y])
        trait_preds = preds[trait_test_mask]
        trait_pred_counts = torch.bincount(trait_preds, minlength=10)
        print(f"Predictions for trait inputs (0-4): {trait_pred_counts.cpu().tolist()}")

        # Per-class accuracy
        for c in range(10):
            mask = test_y == c
            acc = (preds[mask] == c).float().mean().item()
            print(f"  digit {c}: {acc*100:.1f}%")

        # Check logit magnitudes for trait vs non-trait
        trait_logits = logits[trait_test_mask]
        mean_logits = trait_logits.mean(0)
        print(f"Mean logits for trait inputs: {mean_logits.cpu().numpy().round(2)}")

if __name__ == "__main__":
    main()
