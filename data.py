import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from formal_world import (
    sample_semantics,
    generate_term_from_semantics,
    term_semantics,
    term_to_tokens,
)

# ======================
# Token vocabulary & encoding
# ======================

TOK_PAD = "<PAD>"
TOK_CLS = "<CLS>"
TOK_SEP = "<SEP>"

base_tokens = ["a", "b", "c", "d", "0", "+", "(", ")"]
itos = [TOK_PAD, TOK_CLS, TOK_SEP] + base_tokens
stoi = {t: i for i, t in enumerate(itos)}
vocab_size = len(itos)


def encode_pair_terms(t1, t2, max_seq_len: int) -> List[int]:
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


# ======================
# Build pool of terms and pairs
# ======================

def build_term_pool(num_terms: int = 2000, max_atoms: int = 4):
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


def build_pairs(terms, sems, num_pos: int = 3000, num_neg: int = 3000):
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
        raise ValueError(
            "No semantic group has size >= 2; increase num_terms or adjust generator."
        )

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


def build_dataset(
    num_terms: int = 2500,
    max_atoms: int = 4,
    num_pos: int = 4000,
    num_neg: int = 4000,
):
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


def split_dataset(X, y, train_frac: float = 0.7, val_frac: float = 0.15):
    """
    Convenience to split X, y into train/val/test TensorDatasets.
    """
    N = X.size(0)
    indices = np.random.permutation(N)
    n_train = int(train_frac * N)
    n_val = int(val_frac * N)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    return train_ds, val_ds, test_ds
