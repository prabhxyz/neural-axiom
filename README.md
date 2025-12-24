# Neural Axiom Explorer

Learning algebraic structure and theoremhood from data

### Purpose

This project explores whether a neural network can internalize the content of mathematical axioms purely from examples, without being explicitly told the rules. Instead of hard-coding associativity, commutativity, or identity, the model is trained only on labeled pairs of symbolic expressions indicating whether they are mathematically equal. The core goal is to study axiom learning as an invariance learning problem, and to use the learned representation to propose new valid equalities that function as discovered theorems.

At a high level, the system treats mathematics not as symbolic rule execution, but as a distribution over expressions where true equalities form a structured manifold. The model’s job is to learn that manifold.

---

### Theoretical Motivation

In formal mathematics, axioms define equivalence classes over symbolic expressions. For example, in a commutative monoid, expressions like
`a + (b + c)` and `(c + a) + b` are syntactically different but semantically identical. Traditional theorem provers encode this equivalence through rewrite rules and symbolic normalization.

This project instead frames equality as a binary classification problem:

> Given two symbolic expressions, decide whether they are equal under an unknown but consistent set of axioms.

From a learning perspective, this forces the model to discover invariances. Parenthesization must become irrelevant, operand order must collapse, and identity elements must be ignored. The only way to generalize correctly is to internally represent expressions in a way that is invariant under the hidden algebraic symmetries.

This setup is closely related to representation learning under group actions, neural relational reasoning, and invariant/equivariant learning, but applied in a strictly symbolic domain.

---

### Formal System

The underlying mathematical structure is intentionally minimal:

* Variables: a, b, c, d
* Constant: 0
* Binary operation: +

The hidden axioms correspond to a commutative monoid:

* Associativity
* Commutativity
* Identity element

Semantically, every expression corresponds to a multiset of variables. Two expressions are equal if and only if they induce the same multiset. This semantic model is used only to generate labels and to verify discovered theorems, never exposed to the neural network itself.

---

### Learning Setup

The model receives pairs of expressions, linearized as token sequences with parentheses, operators, variables, and special separator tokens. Each pair is mapped to a probability that the two expressions are mathematically equal.

A transformer encoder processes the tokenized pair jointly, allowing attention to model structural correspondences between subexpressions. A pooled representation is passed through a classifier head that outputs a single logit for equivalence.

Crucially, the network never receives explicit rules, rewrite systems, or normal forms. All algebraic structure must be inferred statistically from many examples of equal and non-equal expression pairs.

---

### Theorem Discovery

Once trained, the model acts as a neural theorem proposer. Random expression pairs are generated and scored by the network. Pairs with high predicted probability of equivalence are then passed to a symbolic verifier based on the hidden semantic model.

High-confidence, non-trivial equalities that pass verification are treated as discovered theorems. This creates a neural-symbolic loop where:

1. The neural model proposes candidate identities.
2. A symbolic checker validates correctness.
3. Valid identities can be collected, filtered, or further analyzed.

This mirrors how neural guidance is used in modern automated theorem proving, but with the twist that the neural model is also implicitly learning the axiom system itself.

---

### Why This Matters

This project is a small but concrete step toward models that learn mathematical structure rather than merely imitate proofs. It demonstrates that even simple transformers can recover deep algebraic invariances from data alone, and that theorem discovery can emerge naturally from probabilistic equivalence scoring.

The framework is intentionally extensible. Richer algebraic systems, proof-level supervision, latent axiom representations, or language model heads for explanation can all be layered on top of this core idea.
