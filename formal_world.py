import random
from typing import List, Tuple

VAR_NAMES = ["a", "b", "c", "d"]
N_VARS = len(VAR_NAMES)


def sample_semantics(max_atoms: int = 4) -> Tuple[int, ...]:
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


def generate_term_from_semantics(counts: Tuple[int, ...], max_extra_zeros: int = 2):
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
