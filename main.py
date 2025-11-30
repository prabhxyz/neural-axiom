"""
Neural Axiom Explorer (toy):
- Underlying structure: commutative monoid (a,b,c,d,0,+).
- Hidden axioms: associativity, commutativity, identity.
- Neural model learns to decide if E1 = E2 is true under those axioms.
- Then we use it to propose new equalities (candidate theorems).
"""

import random
from collections import defaultdict
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ======================
# 1. Formal world setup
# ======================

VAR_NAMES = ["a", "b", "c", "d"]
N_VARS = len(VAR_NAMES)

def sample_semantics(max_atoms=4) -> Tuple[int, ...]:
    """
    Sample a random 'semantic vector' = multiset of variables.
    counts[i] = how many times VAR_NAMES[i] appears in the term.
    Total atoms (non-zero vars) <= max_atoms.
    """
    k = random.randint(0, max_atoms)  # number of non-zero atoms
    counts = [0] * N_VARS
    for _ in range(k):
        idx = random.randrange(N_VARS)
        counts[idx] += 1
    return tuple(counts)

def generate_term_from_semantics(counts: Tuple[int, ...], max_extra_zeros=2):
    """
    Build a random binary + tree whose semantics matches 'counts'.
    Optionally sprinkle in some 'zero' leaves that don't change semantics.
    Term representation:
      ("zero",)
      ("var", "a")
      ("plus", left_term, right_term)
    """
    leaves = []
    for i, c in enumerate(counts):
        for _ in range(c):
            leaves.append(("var", VAR_NAMES[i]))
    # Optionally add zeros that won't change semantics
    extra = random.randint(0, max_extra_zeros)
    for _ in range(extra):
        leaves.append(("zero",))
    if not leaves:
        # pure zero term
        return ("zero",)
    random.shuffle(leaves)
    while len(leaves) > 1:
        i = random.randrange(len(leaves))
        a = leaves.pop(i)
        j = random.randrange(len(leaves))
        b = leaves.pop(j)
        leaves.append(("plus", a, b))
    return leaves[0]

def term_semantics(term) -> Tuple[int, ...]:
    """
    Compute the 'semantic vector' of a term = counts of each variable.
    This is the ground truth used to test if two terms are equal.
    """
    kind = term[0]
    if kind == "zero":
        return tuple([0] * N_VARS)
    if kind == "var":
        vec = [0] * N_VARS
        idx = VAR_NAMES.index(term[1])
        vec[idx] = 1
        return tuple(vec)
    if kind == "plus":
        left = term_semantics(term[1])
        right = term_semantics(term[2])
        return tuple(l + r for l, r in zip(left, right))
    raise ValueError(f"Unknown term node kind: {kind}")

def term_to_tokens(term) -> List[str]:
    """
    Turn a term tree into an infix token sequence with parentheses.
    """
    kind = term[0]
    if kind == "zero":
        return ["0"]
    if kind == "var":
        return [term[1]]
    if kind == "plus":
        left = term_to_tokens(term[1])
        right = term_to_tokens(term[2])
        return ["("] + left + ["+"] + right + [")"]
    raise ValueError(f"Unknown term node kind: {kind}")

def term_to_string(term) -> str:
    return " ".join(term_to_tokens(term))


# ======================
# 2. Build pool of terms and pairs
# ======================

def build_term_pool(num_terms=2000, max_atoms=4):
    """
    Generate a bunch of random terms and their semantics.
    """
    terms = []
    sems = []
    for _ in range(num_terms):
        counts = sample_semantics(max_atoms)
        term = generate_term_from_semantics(counts)
        terms.append(term)
        sems.append(term_semantics(term))
    return terms, sems

def group_by_semantics(terms, sems):
    groups = defaultdict(list)
    for idx, s in enumerate(sems):
        groups[s].append(idx)
    return groups

def build_pairs(terms, sems, num_pos=3000, num_neg=3000):
    """
    Build pairs of indices (i,j) plus labels:
      label=1 if same semantics (equivalent under axioms),
      label=0 otherwise.
    """
    groups = group_by_semantics(terms, sems)
    # Positive pairs: same semantics
    pos_pairs = []
    same_keys = [s for s, idxs in groups.items() if len(idxs) >= 2]
    if not same_keys:
        raise ValueError("No semantic group has size >= 2; increase num_terms or adjust generator.")

    while len(pos_pairs) < num_pos:
        s = random.choice(same_keys)
        idxs = groups[s]
        i, j = random.sample(idxs, 2)
        pos_pairs.append((i, j))
        if len(pos_pairs) > num_pos * 2:
            break
    pos_pairs = pos_pairs[:num_pos]

    # Negative pairs: different semantics
    sem_keys = list(groups.keys())
    neg_pairs = []
    while len(neg_pairs) < num_neg:
        s1, s2 = random.sample(sem_keys, 2)
        i = random.choice(groups[s1])
        j = random.choice(groups[s2])
        neg_pairs.append((i, j))
        if len(neg_pairs) > num_neg * 2:
            break
    neg_pairs = neg_pairs[:num_neg]

    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    pairs = pos_pairs + neg_pairs
    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    pairs, labels = zip(*combined)
    return list(pairs), list(labels)


# ======================
# 3. Token vocabulary & encoding
# ======================

TOK_PAD = "<PAD>"
TOK_CLS = "<CLS>"
TOK_SEP = "<SEP>"

base_tokens = ["a", "b", "c", "d", "0", "+", "(", ")"]
itos = [TOK_PAD, TOK_CLS, TOK_SEP] + base_tokens
stoi = {t: i for i, t in enumerate(itos)}
vocab_size = len(itos)

def encode_pair_terms(t1, t2, max_seq_len):
    """
    Encode a pair of terms (t1, t2) as a sequence of token IDs:
    <CLS> t1_tokens <SEP> t2_tokens [PAD...]
    """
    tokens1 = term_to_tokens(t1)
    tokens2 = term_to_tokens(t2)
    tokens = [TOK_CLS] + tokens1 + [TOK_SEP] + tokens2
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    tokens = tokens + [TOK_PAD] * (max_seq_len - len(tokens))
    return [stoi[t] for t in tokens]


def build_dataset(num_terms=2500, max_atoms=4, num_pos=4000, num_neg=4000):
    """
    Returns:
      terms, sems, X (pairs -> token ids), y (labels), max_seq_len
    """
    terms, sems = build_term_pool(num_terms=num_terms, max_atoms=max_atoms)

    # Determine max token length of single term
    max_len_term = max(len(term_to_tokens(t)) for t in terms)
    max_seq_len = 2 * max_len_term + 2  # CLS + term1 + SEP + term2

    pairs, labels = build_pairs(terms, sems, num_pos=num_pos, num_neg=num_neg)

    X = []
    for (i, j) in pairs:
        t1 = terms[i]
        t2 = terms[j]
        X.append(encode_pair_terms(t1, t2, max_seq_len))
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

    return terms, sems, X, y, max_seq_len


# ======================
# 4. Model: transformer-based equivalence classifier
# ======================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class AxiomEquivModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model=64, n_heads=4, num_layers=2, dim_ff=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff) for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, 1)  # logit for "equivalent"

    def forward(self, x):
        # x: (batch, seq_len)
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h)
        cls_repr = h[:, 0, :]          # CLS token embedding
        logit = self.out(cls_repr)     # (batch, 1)
        return logit


# ======================
# 5. Train / validate / test
# ======================

def train_equiv_model():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    terms, sems, X, y, max_seq_len = build_dataset(
        num_terms=2500,  # how many base terms to sample
        max_atoms=4,     # max number of non-zero variables in each term
        num_pos=4000,
        num_neg=4000
    )

    N = X.size(0)
    indices = np.random.permutation(N)
    n_train = int(0.7 * N)
    n_val = int(0.15 * N)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

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

    return model, max_seq_len, (train_losses, val_losses, train_accs, val_accs), (all_probs, all_labels)


# ======================
# 6. Theorem discovery: use the trained model
# ======================

def discover_theorems(model, max_seq_len, num_samples=2000, prob_threshold=0.9, max_atoms=4, device=None):
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


# ======================
# 7. Run everything and plot
# ======================

if __name__ == "__main__":
    model, max_seq_len, (train_losses, val_losses, train_accs, val_accs), (probs, labels) = train_equiv_model()

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
