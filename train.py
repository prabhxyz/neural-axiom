import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data import (
    build_dataset,
    split_dataset,
    encode_pair_terms,
    vocab_size,
)
from formal_world import (
    sample_semantics,
    generate_term_from_semantics,
    term_semantics,
    term_to_string,
)
from model import AxiomEquivModel


def train_equiv_model():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    terms, sems, X, y, max_seq_len = build_dataset(
        num_terms=2500,  # how many base terms to sample
        max_atoms=4,     # max number of non-zero variables in each term
        num_pos=4000,
        num_neg=4000,
    )

    train_ds, val_ds, test_ds = split_dataset(X, y, train_frac=0.7, val_frac=0.15)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = AxiomEquivModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dim_ff=128,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 6
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        avg_train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_total_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_total_loss += loss.item() * xb.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        avg_val_loss = val_total_loss / val_total
        val_acc = val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.3f}"
        )

    # Evaluate on test set
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    test_pred = (all_probs > 0.5).astype(np.float32)
    test_acc = (test_pred == all_labels).mean()
    print(f"Test accuracy: {test_acc:.3f}")

    return model, max_seq_len, (train_losses, val_losses, train_accs, val_accs), (
        all_probs,
        all_labels,
    )


def discover_theorems(
    model,
    max_seq_len: int,
    num_samples: int = 2000,
    prob_threshold: float = 0.9,
    max_atoms: int = 4,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device

    candidates = []
    with torch.no_grad():
        for _ in range(num_samples):
            # Sample two random terms independently
            counts1 = sample_semantics(max_atoms)
            counts2 = sample_semantics(max_atoms)
            t1 = generate_term_from_semantics(counts1)
            t2 = generate_term_from_semantics(counts2)

            # Encode for the model
            ids = encode_pair_terms(t1, t2, max_seq_len)
            xb = torch.tensor([ids], dtype=torch.long, device=device)
            logit = model(xb)
            prob = torch.sigmoid(logit)[0, 0].item()

            # Ground truth equality via semantics
            equal = (term_semantics(t1) == term_semantics(t2))
            candidates.append((t1, t2, prob, equal))

    # Keep those the model is very confident about
    candidates.sort(key=lambda x: x[2], reverse=True)
    discovered = [
        (t1, t2, prob)
        for (t1, t2, prob, equal) in candidates
        if equal and prob >= prob_threshold and term_to_string(t1) != term_to_string(t2)
    ]

    print(f"\nTop discovered equalities (model prob >= {prob_threshold}):")
    for t1, t2, prob in discovered[:10]:
        print(f"  {term_to_string(t1)}  ==  {term_to_string(t2)}   (p={prob:.3f})")

    if not discovered:
        print("  (No high-confidence non-trivial equalities found.)")


if __name__ == "__main__":
    model, max_seq_len, (train_losses, val_losses, train_accs, val_accs), (
        probs,
        labels,
    ) = train_equiv_model()

    epochs = range(1, len(train_losses) + 1)

    # Plot 1: training vs validation loss
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("Training and validation loss (equivalence classifier)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # This graph shows whether the model is learning the invariances induced by the axioms.

    # Plot 2: training vs validation accuracy
    plt.figure()
    plt.plot(epochs, train_accs, label="Train accuracy")
    plt.plot(epochs, val_accs, label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # This graph shows how well the model is predicting theorem vs non-theorem.

    # Plot 3: distribution of predicted probabilities on the test set
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    plt.figure()
    plt.hist(pos_probs, bins=20, alpha=0.5, label="Equal (theorem)")
    plt.hist(neg_probs, bins=20, alpha=0.5, label="Not equal (non-theorem)")
    plt.xlabel("Predicted probability of equivalence")
    plt.ylabel("Count")
    plt.title("Test-set probability distributions")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Ideally, the 'Equal' histogram is concentrated near 1 and 'Not equal' near 0.

    # Finally, use the model to propose some new equalities and verify them
    discover_theorems(model, max_seq_len, num_samples=2000, prob_threshold=0.9)
