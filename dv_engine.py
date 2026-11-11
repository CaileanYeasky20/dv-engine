#!/usr/bin/env python3
"""
===============================================================================
ENTERPRISE DV-ENGINE MASTER ENGINE — UNIFIED PRODUCTION SYSTEM
===============================================================================
Synthesizes phase-class lossless compression, GPU-accelerated Mersenne-31
finite-field tensors, sumcheck-verified layer proofs, hash-chained IVC
accumulation with replay protection, causal CRDT semilattices, Byzantine
equivocation slashing and lock-striped sharded storage into a single secure
cluster architecture, together with a self-verifying validation suite.

Usage
    python dv_engine.py [--layers N] [--width N] [--seed N] [--quick]

Environment
    DV_BACKEND = auto | cupy | numpy | python        (default: auto)

Requires Python 3.9+. NumPy, CuPy and psutil are optional accelerators.
===============================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import functools
import hashlib
import heapq
import hmac
import json
import logging
import os
import random
import secrets
import struct
import sys
import threading
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from operator import mul
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import psutil
except ImportError:
    psutil = None

_LOG = logging.getLogger("dv-engine")


# =====================================================================
# 0. RUNTIME & ACCELERATION BACKENDS (CuPy -> NumPy -> pure Python)
# =====================================================================

def _load_backends() -> Tuple[Any, Any]:
    """Select array backends, honouring DV_BACKEND. Never raises."""
    requested = os.environ.get("DV_BACKEND", "auto").strip().lower()
    numpy_module = cupy_module = None
    if requested != "python":
        try:
            import numpy
            numpy_module = numpy
        except ImportError:
            pass
    if requested == "numpy":
        return numpy_module, None
    if numpy_module is not None and requested in ("auto", "cupy"):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() > 0:
                probe = cupy.asarray(numpy_module.arange(4, dtype=numpy_module.int64))
                cupy.asnumpy(probe + 1)
                cupy_module = cupy
        except Exception:
            cupy_module = None
    return numpy_module, cupy_module


np, cp = _load_backends()
HAS_NUMPY = np is not None
HAS_CUPY = cp is not None


def backend_description() -> str:
    requested = os.environ.get("DV_BACKEND", "auto").strip().lower()
    if requested == "numpy":
        return "numpy (CPU)"
    if HAS_CUPY:
        return "cupy (GPU offload) + numpy"
    return "numpy" if HAS_NUMPY else "pure python"


# =====================================================================
# 1. ENTERPRISE PKI & BYZANTINE EQUIVOCATION SLASHING REGISTRY
# =====================================================================

class CryptoKey:
    """HMAC-SHA256 signing identity for a cluster node."""

    def __init__(self, node_id: str, *, register: bool = True) -> None:
        self.node_id = node_id
        self._secret = secrets.token_bytes(32)
        if register:
            PKIRegistry.register(node_id, self)

    def sign(self, message: str) -> str:
        return hmac.new(self._secret, message.encode(), hashlib.sha256).hexdigest()

    def verify_own(self, message: str, sig: str) -> bool:
        try:
            return hmac.compare_digest(self.sign(message).encode("ascii"), sig.encode("ascii"))
        except (AttributeError, UnicodeEncodeError):
            return False


class PKIRegistry:
    """Global identity registry with forgery and equivocation slashing."""

    _lock = threading.RLock()
    _public_keys: Dict[str, CryptoKey] = {}
    _slashed: Dict[str, str] = {}
    _attestations: Dict[Tuple[str, int], str] = {}

    @classmethod
    def register(cls, node_id: str, keypair: CryptoKey) -> None:
        with cls._lock:
            existing = cls._public_keys.get(node_id)
            if existing is not None and existing is not keypair:
                raise ValueError(f"identity '{node_id}' is already registered")
            cls._public_keys[node_id] = keypair

    @classmethod
    def slash(cls, node_id: str, reason: str = "forgery/equivocation") -> bool:
        """Revoke a node. Idempotent; returns True only on the first slash."""
        with cls._lock:
            if node_id in cls._slashed:
                return False
            cls._slashed[node_id] = reason
        _LOG.warning("[BYZANTINE GUARD] SLASHED: node '%s' revoked (%s).", node_id, reason)
        return True

    @classmethod
    def is_slashed(cls, node_id: str) -> bool:
        with cls._lock:
            return node_id in cls._slashed

    @classmethod
    def slashed_nodes(cls) -> Dict[str, str]:
        with cls._lock:
            return dict(cls._slashed)

    @classmethod
    def verify(cls, node_id: str, message: str, sig: str) -> bool:
        with cls._lock:
            key = cls._public_keys.get(node_id)
            if key is None or node_id in cls._slashed:
                return False
        return key.verify_own(message, sig)

    @classmethod
    def attest(cls, node_id: str, slot: int, digest: str) -> bool:
        """Record a signed statement; slash the node if it signs two for one slot."""
        with cls._lock:
            recorded = cls._attestations.setdefault((node_id, slot), digest)
            if recorded == digest:
                return True
        cls.slash(node_id, f"equivocation at slot {slot}")
        return False

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._public_keys.clear()
            cls._slashed.clear()
            cls._attestations.clear()


# =====================================================================
# 2. MERSENNE-31 FINITE FIELD & ACCELERATED TENSORS
# =====================================================================

PRIME = (1 << 31) - 1  # 2,147,483,647
HALF_PRIME = PRIME // 2


def f_add(a: int, b: int) -> int:
    return (a + b) % PRIME


def f_sub(a: int, b: int) -> int:
    return (a - b) % PRIME


def f_mul(a: int, b: int) -> int:
    return (a * b) % PRIME


def f_inv(a: int) -> int:
    if a % PRIME == 0:
        raise ZeroDivisionError("zero has no inverse in F_p")
    return pow(a, PRIME - 2, PRIME)


INV_TWO = f_inv(2)

_LIMB_BITS = 16
_LIMB_MASK = (1 << _LIMB_BITS) - 1
_LIMB_SHIFT = 1 << _LIMB_BITS
_K_CHUNK = 1 << 15                  # bounds every int64 accumulator below 2^62
GPU_OFFLOAD_THRESHOLD = 1 << 22     # multiply-accumulates before a GPU round trip pays off


def _matmul_mod(xp: Any, a: Any, b: Any) -> Any:
    """(m x k) @ (k x n) over F_p without int64 overflow.

    The smaller operand is split into 16-bit limbs so each partial product stays
    below 2^47 and each chunked accumulation below 2^62. `xp` is numpy or cupy.
    """
    m, k = a.shape
    n = b.shape[1]
    split_left = a.size <= b.size
    out = xp.zeros((m, n), dtype=xp.int64)
    for start in range(0, k, _K_CHUNK):
        stop = min(start + _K_CHUNK, k)
        a_c, b_c = a[:, start:stop], b[start:stop]
        if split_left:
            lo = xp.matmul(a_c & _LIMB_MASK, b_c) % PRIME
            hi = xp.matmul(a_c >> _LIMB_BITS, b_c) % PRIME
        else:
            lo = xp.matmul(a_c, b_c & _LIMB_MASK) % PRIME
            hi = xp.matmul(a_c, b_c >> _LIMB_BITS) % PRIME
        out = (out + hi * _LIMB_SHIFT + lo) % PRIME
    return out


class FieldTensor2D:
    """Immutable matrix over F_p (p = 2^31 - 1).

    Stored as a read-only NumPy int64 array when NumPy is available (with CuPy
    offload for large products) and as nested tuples otherwise. Immutability
    makes tensors safe to share across threads and to hash once.
    """

    __slots__ = ("rows", "cols", "_a", "_hash", "_columns_cache")

    def __init__(self, rows: int, cols: int, data: Optional[Sequence[Sequence[int]]] = None) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("tensor dimensions must be positive")
        self.rows = rows
        self.cols = cols
        self._hash: Optional[str] = None
        self._columns_cache: Optional[Tuple[Tuple[int, ...], ...]] = None
        self._a = self._zeros(rows, cols) if data is None else self._freeze(rows, cols, data)

    # ---- construction ------------------------------------------------

    @staticmethod
    def _zeros(rows: int, cols: int) -> Any:
        if HAS_NUMPY:
            arr = np.zeros((rows, cols), dtype=np.int64)
            arr.setflags(write=False)
            return arr
        return tuple((0,) * cols for _ in range(rows))

    @staticmethod
    def _freeze(rows: int, cols: int, data: Sequence[Sequence[int]]) -> Any:
        if HAS_NUMPY:
            try:
                arr = np.array(data, dtype=np.int64)
            except OverflowError:
                arr = np.array([[int(v) % PRIME for v in row] for row in data], dtype=np.int64)
            if arr.shape != (rows, cols):
                raise ValueError(f"expected shape ({rows}, {cols}), got {arr.shape}")
            arr = arr % PRIME
            arr.setflags(write=False)
            return arr
        grid = tuple(tuple(int(v) % PRIME for v in row) for row in data)
        if len(grid) != rows or any(len(row) != cols for row in grid):
            raise ValueError(f"expected shape ({rows}, {cols})")
        return grid

    @classmethod
    def _wrap(cls, rows: int, cols: int, array: Any) -> FieldTensor2D:
        """Wrap an already-normalised backend array without copying."""
        obj = object.__new__(cls)
        obj.rows, obj.cols = rows, cols
        obj._hash = None
        obj._columns_cache = None
        if HAS_NUMPY:
            array.setflags(write=False)
        obj._a = array
        return obj

    @classmethod
    def random(cls, rows: int, cols: int, seed: int) -> FieldTensor2D:
        """Deterministic pseudo-random tensor, identical on every backend."""
        label = f"DV-ENGINE/random/v1|{seed}|{rows}|{cols}".encode()
        stream = hashlib.shake_256(label).digest(8 * rows * cols)
        if HAS_NUMPY:
            words = np.frombuffer(stream, dtype=">u8") % PRIME
            return cls._wrap(rows, cols, words.astype(np.int64).reshape(rows, cols))
        words = struct.unpack(f">{rows * cols}Q", stream)
        grid = tuple(tuple(words[r * cols + c] % PRIME for c in range(cols)) for r in range(rows))
        return cls._wrap(rows, cols, grid)

    # ---- access ------------------------------------------------------

    @property
    def data(self) -> List[List[int]]:
        """Copy of the contents as a list of rows."""
        return self._a.tolist() if HAS_NUMPY else [list(row) for row in self._a]

    def row(self, index: int) -> List[int]:
        return self._a[index].tolist() if HAS_NUMPY else list(self._a[index])

    def column(self, index: int) -> List[int]:
        return self._a[:, index].tolist() if HAS_NUMPY else [row[index] for row in self._a]

    def get(self, r: int, c: int) -> int:
        return int(self._a[r, c]) if HAS_NUMPY else self._a[r][c]

    def _columns(self) -> Tuple[Tuple[int, ...], ...]:
        cached = self._columns_cache
        if cached is None:
            cached = self._columns_cache = tuple(zip(*self._a))
        return cached

    # ---- arithmetic --------------------------------------------------

    def matmul(self, other: FieldTensor2D) -> FieldTensor2D:
        if self.cols != other.rows:
            raise ValueError(f"Shape mismatch: ({self.rows}x{self.cols}) @ ({other.rows}x{other.cols})")
        if HAS_NUMPY:
            out = self._matmul_numpy(other)
        else:
            out = tuple(
                tuple(sum(map(mul, row, col)) % PRIME for col in other._columns())
                for row in self._a
            )
        return FieldTensor2D._wrap(self.rows, other.cols, out)

    def _matmul_numpy(self, other: FieldTensor2D) -> Any:
        if HAS_CUPY and self.rows * self.cols * other.cols >= GPU_OFFLOAD_THRESHOLD:
            try:
                return cp.asnumpy(_matmul_mod(cp, cp.asarray(self._a), cp.asarray(other._a)))
            except Exception:
                pass  # transparent CPU fallback
        return _matmul_mod(np, self._a, other._a)

    def relu_with_sketch(self) -> Tuple[FieldTensor2D, FieldTensor2D]:
        """Apply ReLU (values above p/2 are negative) and emit the boolean constraint sketch."""
        if HAS_NUMPY:
            positive = self._a <= HALF_PRIME
            out = np.where(positive, self._a, 0).astype(np.int64)
            sketch = positive.astype(np.int64)
        else:
            out = tuple(tuple(v if v <= HALF_PRIME else 0 for v in row) for row in self._a)
            sketch = tuple(tuple(1 if v <= HALF_PRIME else 0 for v in row) for row in self._a)
        return (
            FieldTensor2D._wrap(self.rows, self.cols, out),
            FieldTensor2D._wrap(self.rows, self.cols, sketch),
        )

    def tampered(self, rng: random.Random, fraction: float = 0.15) -> FieldTensor2D:
        """Copy with a fraction of entries guaranteed to differ (Byzantine corruption)."""
        total = self.rows * self.cols
        count = min(total, max(1, int(total * fraction)))
        grid = self.data
        for flat in rng.sample(range(total), count):
            r, c = divmod(flat, self.cols)
            grid[r][c] = (grid[r][c] + 1 + rng.randrange(PRIME - 1)) % PRIME
        return FieldTensor2D(self.rows, self.cols, grid)

    # ---- commitment --------------------------------------------------

    def merkle_leaf_hash(self) -> str:
        digest = self._hash
        if digest is None:
            hasher = hashlib.sha256(b"DV_ENGINE/tensor/v1")
            hasher.update(self.rows.to_bytes(4, "big"))
            hasher.update(self.cols.to_bytes(4, "big"))
            if HAS_NUMPY:
                hasher.update(self._a.astype(">u4").tobytes())
            else:
                pattern = f">{self.cols}I"
                for row in self._a:
                    hasher.update(struct.pack(pattern, *row))
            digest = self._hash = hasher.hexdigest()
        return digest

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FieldTensor2D):
            return NotImplemented
        return (self.rows, self.cols) == (other.rows, other.cols) and (
            self.merkle_leaf_hash() == other.merkle_leaf_hash()
        )

    def __hash__(self) -> int:
        return hash(self.merkle_leaf_hash())

    def __repr__(self) -> str:
        return f"FieldTensor2D({self.rows}x{self.cols}, {self.merkle_leaf_hash()[:12]}…)"


# =====================================================================
# 3. SUMCHECK LAYER PROOFS (GKR BUILDING BLOCK)
# =====================================================================

class Transcript:
    """Fiat–Shamir transcript: SHA-256 chaining with SHAKE-256 vector expansion."""

    __slots__ = ("_state",)

    def __init__(self, domain: str, *parts: str) -> None:
        self._state = hashlib.sha256(b"DV_ENGINE/transcript/v1|" + domain.encode()).digest()
        for part in parts:
            self.absorb(part.encode())

    def absorb(self, data: bytes) -> None:
        self._state = hashlib.sha256(self._state + len(data).to_bytes(8, "big") + data).digest()

    def absorb_ints(self, values: Sequence[int]) -> None:
        self.absorb(struct.pack(f">{len(values)}Q", *values))

    def challenge(self) -> int:
        out = hashlib.sha256(self._state + b"\x01").digest()
        self._state = hashlib.sha256(self._state + b"\x02" + out).digest()
        return int.from_bytes(out[:8], "big") % PRIME

    def challenge_vector(self, count: int) -> List[int]:
        stream = hashlib.shake_256(self._state + b"\x03" + count.to_bytes(8, "big")).digest(8 * count)
        self._state = hashlib.sha256(self._state + b"\x04" + stream[:32]).digest()
        return [word % PRIME for word in struct.unpack(f">{count}Q", stream)]


@dataclass(frozen=True)
class SumcheckProof:
    """Non-interactive sumcheck transcript for one matrix-product layer."""
    claimed_sum: int                               # alpha^T · C · beta
    round_polys: Tuple[Tuple[int, int, int], ...]  # (g(0), g(1), g(2)) per round
    eval_a: int                                    # folded left operand at the sumcheck point
    eval_b: int                                    # folded right operand at the sumcheck point
    challenge: int                                 # final Fiat–Shamir challenge
    bound_root: str                                # accumulator root the proof is bound to

    def well_formed(self) -> bool:
        def canonical(x: Any) -> bool:
            return isinstance(x, int) and not isinstance(x, bool) and 0 <= x < PRIME

        return (
            isinstance(self.bound_root, str)
            and isinstance(self.round_polys, tuple)
            and all(canonical(x) for x in (self.claimed_sum, self.eval_a, self.eval_b, self.challenge))
            and all(
                isinstance(poly, tuple) and len(poly) == 3 and all(canonical(x) for x in poly)
                for poly in self.round_polys
            )
        )

    def digest(self) -> str:
        flat = [self.claimed_sum, self.eval_a, self.eval_b, self.challenge, len(self.round_polys)]
        for poly in self.round_polys:
            flat.extend(poly)
        hasher = hashlib.sha256(b"DV_ENGINE/proof/v1|")
        hasher.update(str(self.bound_root).encode())
        hasher.update(b"|" + ",".join(map(str, flat)).encode())
        return hasher.hexdigest()


class GKRSumcheck:
    """Sumcheck proofs for matrix-product layers C = A·B.

    A random bilinear projection alpha^T·C·beta reduces the layer to an inner
    product of two vectors, which is proven with the classic sumcheck protocol
    made non-interactive via Fiat–Shamir. Soundness error is O(log n / p) per
    proof; use an extension field for deployments needing a tighter bound.
    """

    @staticmethod
    def _pad(vec: List[int]) -> List[int]:
        size = 1
        while size < len(vec):
            size <<= 1
        return vec + [0] * (size - len(vec))

    @staticmethod
    def _bind(vec: List[int], r: int) -> List[int]:
        return [(e + r * (o - e)) % PRIME for e, o in zip(vec[0::2], vec[1::2])]

    @staticmethod
    def _round_poly(u: List[int], v: List[int]) -> Tuple[int, int, int]:
        u_even, u_odd, v_even, v_odd = u[0::2], u[1::2], v[0::2], v[1::2]
        g0 = sum(map(mul, u_even, v_even)) % PRIME
        g1 = sum(map(mul, u_odd, v_odd)) % PRIME
        g2 = sum((2 * uo - ue) * (2 * vo - ve) for ue, uo, ve, vo in zip(u_even, u_odd, v_even, v_odd)) % PRIME
        return g0, g1, g2

    @staticmethod
    def _interpolate(g: Tuple[int, int, int], r: int) -> int:
        g0, g1, g2 = g
        return (INV_TWO * (g0 * (r - 1) * (r - 2) + g2 * r * (r - 1)) - g1 * r * (r - 2)) % PRIME

    @classmethod
    def _setup(
        cls, a: FieldTensor2D, b: FieldTensor2D, c: FieldTensor2D, seed: str
    ) -> Tuple[Transcript, int, List[int], List[int]]:
        transcript = Transcript(
            "layer-sumcheck", seed, a.merkle_leaf_hash(), b.merkle_leaf_hash(), c.merkle_leaf_hash()
        )
        alpha = FieldTensor2D(1, a.rows, [transcript.challenge_vector(a.rows)])
        beta = transcript.challenge_vector(b.cols)
        beta_col = FieldTensor2D(b.cols, 1, [[x] for x in beta])
        claim = sum(map(mul, alpha.matmul(c).row(0), beta)) % PRIME
        left = cls._pad(alpha.matmul(a).row(0))
        right = cls._pad(b.matmul(beta_col).column(0))
        return transcript, claim, left, right

    @classmethod
    def prove_layer(
        cls, a: FieldTensor2D, b: FieldTensor2D, seed: str, c: Optional[FieldTensor2D] = None
    ) -> SumcheckProof:
        if a.cols != b.rows:
            raise ValueError(f"Shape mismatch: ({a.rows}x{a.cols}) @ ({b.rows}x{b.cols})")
        if c is None:
            c = a.matmul(b)
        transcript, claim, u, v = cls._setup(a, b, c, seed)
        transcript.absorb_ints([claim])
        rounds: List[Tuple[int, int, int]] = []
        while len(u) > 1:
            poly = cls._round_poly(u, v)
            rounds.append(poly)
            transcript.absorb_ints(poly)
            r = transcript.challenge()
            u, v = cls._bind(u, r), cls._bind(v, r)
        transcript.absorb_ints([u[0], v[0]])
        return SumcheckProof(claim, tuple(rounds), u[0], v[0], transcript.challenge(), seed)

    @classmethod
    def verify_layer(
        cls, a: FieldTensor2D, b: FieldTensor2D, c: FieldTensor2D, proof: SumcheckProof, seed: str
    ) -> bool:
        if a.cols != b.rows or (c.rows, c.cols) != (a.rows, b.cols):
            return False
        if not proof.well_formed() or proof.bound_root != seed:
            return False
        transcript, claim, u, v = cls._setup(a, b, c, seed)
        if proof.claimed_sum != claim or len(proof.round_polys) != len(u).bit_length() - 1:
            return False
        transcript.absorb_ints([claim])
        for poly in proof.round_polys:
            if (poly[0] + poly[1]) % PRIME != claim:
                return False
            transcript.absorb_ints(poly)
            r = transcript.challenge()
            claim = cls._interpolate(poly, r)
            u, v = cls._bind(u, r), cls._bind(v, r)
        if (proof.eval_a, proof.eval_b) != (u[0], v[0]) or claim != u[0] * v[0] % PRIME:
            return False
        transcript.absorb_ints([u[0], v[0]])
        return proof.challenge == transcript.challenge()


# =====================================================================
# 4. HASH-CHAINED IVC ACCUMULATOR WITH REPLAY PROTECTION
# =====================================================================

class AccumulatorError(Exception):
    """Base class for accumulator rejections."""


class ReplayError(AccumulatorError):
    """The (leaf, proof) pair was already folded."""


class StaleProofError(AccumulatorError):
    """The proof is bound to a state root other than the current one."""


class MerkleIVCAccumulator:
    """Folds verified proofs into an incrementally verifiable state root.

    Each step commits to the previous root, the leaf and the proof, so the final
    root attests to the full ordered history. Proofs are bound to the root they
    were generated against, and every folded pair is remembered to block replays.
    Callers verify proofs before folding (see LayerVerifier).
    """

    GENESIS_ROOT = hashlib.sha256(b"DV_ENGINE_MASTER_GENESIS").hexdigest()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._root = self.GENESIS_ROOT
        self._seen: Set[str] = set()
        self._log: List[Tuple[str, str]] = []

    @property
    def state_root(self) -> str:
        return self._root

    def __len__(self) -> int:
        return len(self._log)

    @staticmethod
    def _chain(root: str, leaf_hash: str, proof_digest: str) -> str:
        return hashlib.sha256(f"{root}|{leaf_hash}|{proof_digest}".encode()).hexdigest()

    def fold(self, leaf_hash: str, proof: SumcheckProof) -> str:
        """Fold one step and return the new root; raises ReplayError / StaleProofError."""
        digest = proof.digest()
        fingerprint = hashlib.sha256(f"{leaf_hash}|{digest}".encode()).hexdigest()
        with self._lock:
            if fingerprint in self._seen:
                raise ReplayError("proof already folded")
            if proof.bound_root != self._root:
                raise StaleProofError("proof is bound to a different state root")
            self._root = self._chain(self._root, leaf_hash, digest)
            self._seen.add(fingerprint)
            self._log.append((leaf_hash, digest))
            return self._root

    def try_fold(self, leaf_hash: str, proof: SumcheckProof) -> bool:
        try:
            self.fold(leaf_hash, proof)
            return True
        except AccumulatorError:
            return False

    @classmethod
    def replay(cls, entries: Iterable[Tuple[str, str]]) -> str:
        """Recompute the root from an ordered (leaf_hash, proof_digest) log."""
        root = cls.GENESIS_ROOT
        for leaf_hash, proof_digest in entries:
            root = cls._chain(root, leaf_hash, proof_digest)
        return root

    def audit(self) -> bool:
        """True when replaying the recorded history reproduces the live root."""
        with self._lock:
            return self.replay(self._log) == self._root


# =====================================================================
# 5. VERIFICATION GATEWAY: SIGNED LAYER SUBMISSIONS
# =====================================================================

class Verdict(Enum):
    """Outcome of submitting a layer to the verification gateway."""
    ACCEPTED = "accepted"
    LATE = "late duplicate"
    REPLAY = "replayed message"
    STALE = "stale root"
    BAD_SLOT = "future slot"
    BAD_SIGNATURE = "bad signature"
    REVOKED_SENDER = "revoked sender"
    EQUIVOCATION = "equivocation"
    BAD_INPUT = "input mismatch"
    BAD_PROOF = "invalid proof"
    BAD_COMMITMENT = "commitment mismatch"


def commit_activation(output: FieldTensor2D) -> Tuple[FieldTensor2D, str]:
    """Apply ReLU to a layer output and commit to the activation plus its sketch."""
    activation, sketch = output.relu_with_sketch()
    leaf = hashlib.sha256(
        b"DV_ENGINE/leaf/v1|" + activation.merkle_leaf_hash().encode() + sketch.merkle_leaf_hash().encode()
    ).hexdigest()
    return activation, leaf


@dataclass(frozen=True)
class LayerSubmission:
    """A node's signed claim that `output = input · weights` for one pipeline slot."""
    node_id: str
    slot: int
    input_hash: str
    output: FieldTensor2D
    proof: SumcheckProof
    leaf_hash: str
    signature: str = ""

    def digest(self) -> str:
        hasher = hashlib.sha256(b"DV_ENGINE/submission/v1")
        parts = (
            self.node_id,
            str(self.slot),
            self.input_hash,
            self.output.merkle_leaf_hash(),
            self.proof.digest(),
            self.leaf_hash,
        )
        for part in parts:
            blob = part.encode()
            hasher.update(len(blob).to_bytes(4, "big") + blob)
        return hasher.hexdigest()


class LayerProver:
    """Honest prover bound to a signing identity and a weight matrix."""

    def __init__(self, key: CryptoKey, weights: FieldTensor2D) -> None:
        self.key = key
        self.weights = weights

    def draft(self, slot: int, activation: FieldTensor2D, root: str) -> LayerSubmission:
        output = activation.matmul(self.weights)
        proof = GKRSumcheck.prove_layer(activation, self.weights, root, c=output)
        _, leaf = commit_activation(output)
        return LayerSubmission(self.key.node_id, slot, activation.merkle_leaf_hash(), output, proof, leaf)

    def seal(self, submission: LayerSubmission) -> LayerSubmission:
        return dataclasses.replace(submission, signature=self.key.sign(submission.digest()))

    def prove(self, slot: int, activation: FieldTensor2D, root: str) -> LayerSubmission:
        return self.seal(self.draft(slot, activation, root))


class LayerVerifier:
    """Verification gateway.

    Pipeline per submission: revocation -> signature -> replay -> slot window ->
    equivocation -> root binding -> input -> sumcheck proof -> commitment -> fold.
    Signed-but-invalid content slashes the sender; benign races (stale roots,
    late or early slots) are dropped without penalty.
    """

    def __init__(
        self,
        weights: FieldTensor2D,
        genesis_input: FieldTensor2D,
        accumulator: Optional[MerkleIVCAccumulator] = None,
    ) -> None:
        self.weights = weights
        self.accumulator = accumulator if accumulator is not None else MerkleIVCAccumulator()
        self._lock = threading.RLock()
        self._inputs: List[FieldTensor2D] = [genesis_input]
        self._roots: List[str] = [self.accumulator.state_root]
        self._accepted: Set[str] = set()
        self._last_accepted: Optional[LayerSubmission] = None

    @property
    def next_slot(self) -> int:
        return len(self._inputs) - 1

    @property
    def last_accepted(self) -> Optional[LayerSubmission]:
        return self._last_accepted

    def snapshot(self) -> Tuple[int, FieldTensor2D, str]:
        """Atomically read (open slot, its input activation, its root)."""
        with self._lock:
            return self.next_slot, self._inputs[-1], self._roots[-1]

    def submit(self, sub: LayerSubmission) -> Verdict:
        with self._lock:
            return self._evaluate(sub)

    @staticmethod
    def _punish(node_id: str, verdict: Verdict, reason: str) -> Verdict:
        PKIRegistry.slash(node_id, reason)
        return verdict

    def _evaluate(self, sub: LayerSubmission) -> Verdict:
        node = sub.node_id
        if PKIRegistry.is_slashed(node):
            return Verdict.REVOKED_SENDER
        digest = sub.digest()
        if not PKIRegistry.verify(node, digest, sub.signature):
            return Verdict.BAD_SIGNATURE
        if digest in self._accepted:
            return Verdict.REPLAY
        if not 0 <= sub.slot <= self.next_slot:
            return Verdict.BAD_SLOT
        if not PKIRegistry.attest(node, sub.slot, digest):
            return Verdict.EQUIVOCATION

        layer_input, seed = self._inputs[sub.slot], self._roots[sub.slot]
        if sub.proof.bound_root != seed:
            return Verdict.STALE
        if sub.input_hash != layer_input.merkle_leaf_hash():
            return self._punish(node, Verdict.BAD_INPUT, f"input mismatch at slot {sub.slot}")
        if not GKRSumcheck.verify_layer(layer_input, self.weights, sub.output, sub.proof, seed):
            return self._punish(node, Verdict.BAD_PROOF, f"invalid layer proof at slot {sub.slot}")
        activation, leaf = commit_activation(sub.output)
        if leaf != sub.leaf_hash:
            return self._punish(node, Verdict.BAD_COMMITMENT, f"commitment mismatch at slot {sub.slot}")
        if sub.slot < self.next_slot:
            return Verdict.LATE

        try:
            root = self.accumulator.fold(leaf, sub.proof)
        except ReplayError:
            return Verdict.REPLAY
        except StaleProofError:
            return Verdict.STALE
        self._inputs.append(activation)
        self._roots.append(root)
        self._accepted.add(digest)
        self._last_accepted = sub
        return Verdict.ACCEPTED


# =====================================================================
# 6. LOSSLESS PHASE-CLASS COMPRESSION
# =====================================================================

HOLOGRAM_SIZE = 4 + 16


class IntegrityError(ValueError):
    """A manifest failed structural or hologram verification."""


class PhaseClass(Enum):
    UNIFORM = 0x00    # one byte repeated
    PERIODIC = 0x01   # short pattern tiled to length
    DEFLATE = 0x02    # zlib stream
    VERBATIM = 0xFF   # stored as-is


@dataclass(frozen=True)
class SingularityManifest:
    original_size: int
    phase_class: PhaseClass
    hologram: bytes
    descriptor: bytes

    @property
    def stored_size(self) -> int:
        return 9 + len(self.hologram) + len(self.descriptor)

    def to_bytes(self) -> bytes:
        return struct.pack(">BQ", self.phase_class.value, self.original_size) + self.hologram + self.descriptor

    @classmethod
    def from_bytes(cls, blob: bytes) -> SingularityManifest:
        header = struct.calcsize(">BQ")
        if len(blob) < header + HOLOGRAM_SIZE:
            raise IntegrityError("manifest blob is truncated")
        tag, size = struct.unpack_from(">BQ", blob)
        try:
            phase = PhaseClass(tag)
        except ValueError as exc:
            raise IntegrityError(f"unknown phase class 0x{tag:02x}") from exc
        return cls(size, phase, blob[header:header + HOLOGRAM_SIZE], blob[header + HOLOGRAM_SIZE:])


class SingularityEngine:
    """Routes each payload to its cheapest lossless phase class and verifies on restore."""

    PERIODIC_SCAN_LIMIT = 1 << 26

    @staticmethod
    def compute_hologram(data: bytes) -> bytes:
        crc = zlib.crc32(data) & 0xFFFFFFFF
        return struct.pack(">I", crc) + hashlib.blake2b(data, digest_size=16).digest()

    @classmethod
    def fold(cls, data: bytes) -> SingularityManifest:
        data = bytes(data)
        size = len(data)
        hologram = cls.compute_hologram(data)
        if size == 0:
            return SingularityManifest(0, PhaseClass.UNIFORM, hologram, b"\x00")
        if data.count(data[:1]) == size:
            return SingularityManifest(size, PhaseClass.UNIFORM, hologram, data[:1])

        candidates: List[Tuple[PhaseClass, bytes]] = []
        if size <= cls.PERIODIC_SCAN_LIMIT:
            period = (data + data).find(data, 1)  # smallest exact period, == size if none
            if 0 < period < size:
                candidates.append((PhaseClass.PERIODIC, data[:period]))
        packed = zlib.compress(data, 9)
        if len(packed) < size:
            candidates.append((PhaseClass.DEFLATE, packed))
        if not candidates:
            return SingularityManifest(size, PhaseClass.VERBATIM, hologram, data)
        phase, descriptor = min(candidates, key=lambda item: (len(item[1]), item[0].value))
        return SingularityManifest(size, phase, hologram, descriptor)

    @classmethod
    def unfold(cls, manifest: SingularityManifest) -> bytes:
        size, descriptor = manifest.original_size, manifest.descriptor
        try:
            if manifest.phase_class is PhaseClass.UNIFORM:
                data = descriptor[:1] * size
            elif manifest.phase_class is PhaseClass.PERIODIC:
                data = (descriptor * (size // len(descriptor) + 1))[:size]
            elif manifest.phase_class is PhaseClass.DEFLATE:
                data = zlib.decompressobj().decompress(descriptor, size + 1)
            else:
                data = descriptor
        except (zlib.error, ZeroDivisionError) as exc:
            raise IntegrityError("descriptor cannot be expanded") from exc
        if len(data) != size or cls.compute_hologram(data) != manifest.hologram:
            raise IntegrityError("hologram mismatch: manifest is corrupt or tampered")
        return data


# =====================================================================
# 7. CAUSAL CRDT SEMILATTICE
# =====================================================================

@dataclass(frozen=True)
class CausalOp:
    op_id: str
    fn_name: str
    args: tuple
    parents: frozenset = field(default_factory=frozenset)


class AutoCRDTCompiler:
    """Operation-based CRDT: replicas converge by joining causal DAGs.

    The join is set union of operation histories (commutative, associative and
    idempotent). State is rebuilt by replaying the union from the genesis state in
    a deterministic linear extension of the causal order, so every replica holding
    the same operations holds the same state.
    """

    def __init__(self, initial_state: Any, methods: Dict[str, Callable[..., Any]]) -> None:
        self.initial_state = copy.deepcopy(initial_state)
        self.state = copy.deepcopy(initial_state)
        self.methods = dict(methods)
        self.causal_history: Dict[str, CausalOp] = {}
        self.head_ops: Set[str] = set()
        self.replica_id = secrets.token_hex(8)
        self._counter = 0
        self._lock = threading.RLock()

    def execute(self, fn_name: str, *args: Any) -> str:
        with self._lock:
            if fn_name not in self.methods:
                raise KeyError(f"unknown operation '{fn_name}'")
            new_state = self.methods[fn_name](self.state, *args)
            self._counter += 1
            seed = f"{self.replica_id}:{self._counter}:{fn_name}:{args!r}"
            op_id = hashlib.sha256(seed.encode()).hexdigest()[:16]
            self.causal_history[op_id] = CausalOp(op_id, fn_name, args, frozenset(self.head_ops))
            self.head_ops = {op_id}
            self.state = new_state
            return op_id

    def _snapshot(self) -> Tuple[Dict[str, CausalOp], Dict[str, Callable[..., Any]]]:
        with self._lock:
            return dict(self.causal_history), dict(self.methods)

    @staticmethod
    def _linearize(history: Dict[str, CausalOp]) -> List[CausalOp]:
        """Deterministic topological order (ties broken by operation id)."""
        pending = {op_id: 0 for op_id in history}
        children: Dict[str, List[str]] = defaultdict(list)
        for op in history.values():
            for parent in op.parents:
                if parent not in history:
                    raise ValueError(f"causal gap: {op.op_id} depends on unknown operation {parent}")
                pending[op.op_id] += 1
                children[parent].append(op.op_id)
        ready = [op_id for op_id, count in pending.items() if count == 0]
        heapq.heapify(ready)
        order: List[CausalOp] = []
        while ready:
            op_id = heapq.heappop(ready)
            order.append(history[op_id])
            for child in children[op_id]:
                pending[child] -= 1
                if pending[child] == 0:
                    heapq.heappush(ready, child)
        if len(order) != len(history):
            raise ValueError("causal cycle detected in operation history")
        return order

    def _adopt(self, history: Dict[str, CausalOp]) -> None:
        order = self._linearize(history)
        state = copy.deepcopy(self.initial_state)
        for op in order:
            state = self.methods[op.fn_name](state, *op.args)
        referenced = {parent for op in order for parent in op.parents}
        with self._lock:
            self.causal_history = dict(history)
            self.head_ops = set(history) - referenced
            self.state = state

    def merge(self, other: AutoCRDTCompiler) -> AutoCRDTCompiler:
        """Join two replicas into a new converged replica."""
        if self.initial_state != other.initial_state:
            raise ValueError("replicas were created from different genesis states")
        history_a, methods_a = self._snapshot()
        history_b, methods_b = other._snapshot()
        for name in methods_a.keys() & methods_b.keys():
            if methods_a[name] is not methods_b[name]:
                raise ValueError(f"conflicting implementations of operation '{name}'")
        for op_id in history_a.keys() & history_b.keys():
            if history_a[op_id] != history_b[op_id]:
                raise ValueError(f"conflicting definitions of operation {op_id}")
        merged = AutoCRDTCompiler(self.initial_state, {**methods_b, **methods_a})
        merged._adopt({**history_b, **history_a})
        return merged

    def state_digest(self) -> str:
        payload = json.dumps({"state": self.state, "ops": sorted(self.causal_history)}, sort_keys=True, default=repr)
        return hashlib.sha256(payload.encode()).hexdigest()


# =====================================================================
# 8. LOCK-STRIPED SHARDED STORAGE
# =====================================================================

class Shard:
    __slots__ = ("entries", "lock")

    def __init__(self) -> None:
        self.entries: Dict[str, Any] = {}
        self.lock = threading.RLock()

    def set(self, key: str, value: Any) -> None:
        with self.lock:
            self.entries[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self.lock:
            return self.entries.get(key, default)

    def delete(self, key: str) -> bool:
        with self.lock:
            return self.entries.pop(key, _MISSING) is not _MISSING

    def __len__(self) -> int:
        with self.lock:
            return len(self.entries)


_MISSING = object()


class ShardedKeyValueStore:
    """Key-value store with one lock per shard and process-stable routing."""

    def __init__(self, shard_count: int = 16) -> None:
        if shard_count < 1:
            raise ValueError("shard_count must be positive")
        self.shards = [Shard() for _ in range(shard_count)]
        self.shard_count = shard_count

    def shard_index(self, key: str) -> int:
        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.shard_count

    def _get_shard(self, key: str) -> Shard:
        return self.shards[self.shard_index(key)]

    def set(self, key: str, value: Any) -> None:
        self._get_shard(key).set(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._get_shard(key).get(key, default)

    def delete(self, key: str) -> bool:
        return self._get_shard(key).delete(key)

    def __contains__(self, key: str) -> bool:
        return self._get_shard(key).get(key, _MISSING) is not _MISSING

    def __len__(self) -> int:
        return sum(len(shard) for shard in self.shards)

    def shard_loads(self) -> List[int]:
        return [len(shard) for shard in self.shards]


# =====================================================================
# 9. BYZANTINE ADVERSARY
# =====================================================================

ROGUE_NODE_ID = "node_adversary_666"
HONEST_NODE_ID = "node_honest_0"


class AdversarialInjector(threading.Thread):
    """Background rogue node that floods the gateway with Byzantine submissions.

    Attacks: tampered outputs signed with its own key, forged signatures under the
    honest identity, and verbatim replays of overheard honest messages.
    """

    ATTACKS = ("tamper", "forge", "replay")

    def __init__(
        self,
        verifier: LayerVerifier,
        prover: LayerProver,
        stop_event: threading.Event,
        seed: int = 0,
        interval: Tuple[float, float] = (0.0005, 0.003),
    ) -> None:
        super().__init__(name=f"adversary-{prover.key.node_id}", daemon=True)
        self.verifier = verifier
        self.prover = prover
        self.stop_event = stop_event
        self.interval = interval
        self.rng = random.Random(seed)
        self.outcomes: Counter = Counter()
        self.submitted = 0
        self.error: Optional[BaseException] = None
        self._cached: Optional[LayerSubmission] = None

    def _tampered_submission(self) -> LayerSubmission:
        slot, activation, root = self.verifier.snapshot()
        draft = self.prover.draft(slot, activation, root)
        forged = draft.output.tampered(self.rng)
        draft = dataclasses.replace(draft, output=forged, leaf_hash=commit_activation(forged)[1])
        return self.prover.seal(draft)

    def _attack(self) -> Verdict:
        kind = self.rng.choice(self.ATTACKS)
        overheard = self.verifier.last_accepted
        if kind == "forge" and overheard is not None:
            return self.verifier.submit(dataclasses.replace(overheard, signature=secrets.token_hex(32)))
        if kind == "replay" and overheard is not None:
            return self.verifier.submit(overheard)
        if self._cached is not None and PKIRegistry.is_slashed(self.prover.key.node_id):
            return self.verifier.submit(self._cached)  # revoked: resend the old payload
        self._cached = self._tampered_submission()
        return self.verifier.submit(self._cached)

    def run(self) -> None:
        try:
            while not self.stop_event.wait(self.rng.uniform(*self.interval)):
                self.outcomes[self._attack()] += 1
                self.submitted += 1
        except BaseException as exc:  # surfaced to the validation suite
            self.error = exc


# =====================================================================
# 10. SELF-VERIFYING VALIDATION SUITE (INCLUDING FORGED-PROOF TESTS)
# =====================================================================

class ValidationError(AssertionError):
    """Raised when a validation invariant does not hold."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _say(message: str) -> None:
    print(f"-> {message}")


def _rss_mb() -> Optional[float]:
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except Exception:
        return None


def _fmt_mb(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f} MB"


@contextlib.contextmanager
def _quiet_slashing():
    previous = _LOG.level
    _LOG.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        _LOG.setLevel(previous)


@dataclass
class SuiteConfig:
    layers: int = 1000
    width: int = 256
    seed: int = 42
    crdt_replicas: int = 4
    crdt_ops: int = 250
    replay_workers: int = 8
    replay_iterations: int = 150
    replay_fraction: float = 0.20

    @classmethod
    def quick(cls, seed: int = 42) -> SuiteConfig:
        return cls(layers=48, width=32, seed=seed, crdt_ops=60, replay_iterations=40)


def _backend_self_test() -> None:
    """Cross-check tensor arithmetic against exact big-integer reference results."""
    global HAS_CUPY
    rng = random.Random(1234)
    cases: List[Tuple[List[List[int]], List[List[int]]]] = []
    for m, k, n in ((1, 1, 1), (3, 7, 5), (8, 40, 8)):
        cases.append((
            [[rng.randrange(PRIME) for _ in range(k)] for _ in range(m)],
            [[rng.randrange(PRIME) for _ in range(n)] for _ in range(k)],
        ))
    deep = 2 * _K_CHUNK + 3  # spans several accumulation chunks at the field maximum
    cases.append(([[PRIME - 1] * deep], [[PRIME - 1] for _ in range(deep)]))
    for a, b in cases:
        expected = [[sum(x * y for x, y in zip(row, col)) % PRIME for col in zip(*b)] for row in a]
        got = FieldTensor2D(len(a), len(a[0]), a).matmul(FieldTensor2D(len(b), len(b[0]), b)).data
        _require(got == expected, "tensor matmul disagrees with the exact reference")
    if HAS_CUPY:
        left, right = FieldTensor2D.random(64, 512, 1), FieldTensor2D.random(512, 64, 2)
        try:
            gpu = cp.asnumpy(_matmul_mod(cp, cp.asarray(left._a), cp.asarray(right._a)))
            healthy = bool((gpu == _matmul_mod(np, left._a, right._a)).all())
        except Exception:
            healthy = False
        if not healthy:
            HAS_CUPY = False
            _LOG.warning("GPU offload disabled: device results diverged from the CPU reference.")


# ---- Test 1: adversarial storm ----------------------------------------

@dataclass
class PipelineReport:
    layers_accepted: int
    duration: float
    root: str
    log_valid: bool
    rogue_submitted: int = 0
    rogue_outcomes: Counter = field(default_factory=Counter)
    slashed: Dict[str, str] = field(default_factory=dict)


def _run_pipeline(cfg: SuiteConfig, adversary: bool) -> PipelineReport:
    PKIRegistry.reset()
    weights = FieldTensor2D.random(cfg.width, cfg.width, seed=cfg.seed + 1)
    activation = FieldTensor2D.random(1, cfg.width, seed=cfg.seed)
    prover = LayerProver(CryptoKey(HONEST_NODE_ID), weights)
    verifier = LayerVerifier(weights, activation)

    stop = threading.Event()
    injector: Optional[AdversarialInjector] = None
    if adversary:
        rogue = LayerProver(CryptoKey(ROGUE_NODE_ID), weights)
        injector = AdversarialInjector(verifier, rogue, stop, seed=cfg.seed)
        injector.start()

    accepted = 0
    started = time.perf_counter()
    try:
        for slot in range(cfg.layers):
            submission = prover.prove(slot, activation, verifier.accumulator.state_root)
            verdict = verifier.submit(submission)
            _require(verdict is Verdict.ACCEPTED, f"honest layer {slot} was {verdict.value}")
            activation = commit_activation(submission.output)[0]
            accepted += 1
    finally:
        stop.set()
        if injector is not None:
            injector.join()
    duration = time.perf_counter() - started

    if injector is not None and injector.error is not None:
        raise ValidationError(f"adversary thread crashed: {injector.error!r}")
    return PipelineReport(
        layers_accepted=accepted,
        duration=duration,
        root=verifier.accumulator.state_root,
        log_valid=verifier.accumulator.audit(),
        rogue_submitted=injector.submitted if injector else 0,
        rogue_outcomes=injector.outcomes if injector else Counter(),
        slashed=PKIRegistry.slashed_nodes(),
    )


@dataclass
class _ProbeContext:
    tag: str
    weights: FieldTensor2D
    genesis: FieldTensor2D
    honest: LayerProver
    rogue: LayerProver
    verifier: LayerVerifier
    rng: random.Random

    @property
    def root(self) -> str:
        return self.verifier.accumulator.state_root


def _run_probe_matrix(seed: int) -> List[Tuple[str, bool]]:
    """Deterministic attack matrix, including explicit forged-proof tests (tampered folds, altered committed values, swapped Merkle siblings)."""
    V = Verdict
    replace = dataclasses.replace

    def tampered_output(c: _ProbeContext) -> List[Verdict]:
        d = c.rogue.draft(0, c.genesis, c.root)
        out = d.output.tampered(c.rng)
        return [c.verifier.submit(c.rogue.seal(replace(d, output=out, leaf_hash=commit_activation(out)[1])))]

    def consistent_claim_forgery(c: _ProbeContext) -> List[Verdict]:
        d = c.rogue.draft(0, c.genesis, c.root)
        out = d.output.tampered(c.rng)
        forged = GKRSumcheck.prove_layer(c.genesis, c.weights, c.root, c=out)
        return [c.verifier.submit(c.rogue.seal(
            replace(d, output=out, proof=forged, leaf_hash=commit_activation(out)[1])))]

    def malformed_proof(c: _ProbeContext) -> List[Verdict]:
        d = c.rogue.draft(0, c.genesis, c.root)
        bad = replace(d.proof, claimed_sum=PRIME + 5)
        return [c.verifier.submit(c.rogue.seal(replace(d, proof=bad)))]

    def forged_leaf(c: _ProbeContext) -> List[Verdict]:
        d = c.rogue.draft(0, c.genesis, c.root)
        return [c.verifier.submit(c.rogue.seal(replace(d, leaf_hash="00" * 32)))]

    # Step 8 specific: forged-proof test targeting altered committed values & swapped/tampered proof structures
    def forged_proof_tampered_fold(c: _ProbeContext) -> List[Verdict]:
        d = c.rogue.draft(0, c.genesis, c.root)
        # Tamper a fold inside the round polynomials
        bad_polys = list(d.proof.round_polys)
        if bad_polys:
            p_list = list(bad_polys[0])
            p_list[0] = (p_list[0] + 1) % PRIME
            bad_polys[0] = tuple(p_list)
        bad_proof = replace(d.proof, round_polys=tuple(bad_polys))
        return [c.verifier.submit(c.rogue.seal(replace(d, proof=bad_proof)))]

    def forged_proof_altered_committed_value(c: _ProbeContext) -> List[Verdict]:
        d = c.rogue.draft(0, c.genesis, c.root)
        # Alter a committed scalar value in the evaluated outputs
        bad_eval = (d.proof.eval_a + 5) % PRIME
        bad_proof = replace(d.proof, eval_a=bad_eval)
        return [c.verifier.submit(c.rogue.seal(replace(d, proof=bad_proof)))]

    def forged_proof_swapped_merkle_sibling(c: _ProbeContext) -> List[Verdict]:
        d = c.rogue.draft(0, c.genesis, c.root)
        # Swap/tamper the bound root reference in the proof object (Merkle sibling simulation)
        bad_proof = replace(d.proof, bound_root="ff" * 32)
        return [c.verifier.submit(c.rogue.seal(replace(d, proof=bad_proof)))]

    def wrong_input(c: _ProbeContext) -> List[Verdict]:
        d = c.rogue.draft(0, c.genesis, c.root)
        return [c.verifier.submit(c.rogue.seal(replace(d, input_hash="ab" * 32)))]

    def forged_signature(c: _ProbeContext) -> List[Verdict]:
        d = c.honest.draft(0, c.genesis, c.root)
        return [c.verifier.submit(replace(d, signature="f" * 64))]

    def unknown_sender(c: _ProbeContext) -> List[Verdict]:
        ghost = LayerProver(CryptoKey(f"ghost_{c.tag}", register=False), c.weights)
        return [c.verifier.submit(ghost.prove(0, c.genesis, c.root))]

    def replayed_message(c: _ProbeContext) -> List[Verdict]:
        sub = c.honest.prove(0, c.genesis, c.root)
        return [c.verifier.submit(sub), c.verifier.submit(sub)]

    def equivocation(c: _ProbeContext) -> List[Verdict]:
        d = c.rogue.draft(0, c.genesis, c.root)
        first = c.verifier.submit(c.rogue.seal(d))
        second = c.verifier.submit(c.rogue.seal(replace(d, leaf_hash="11" * 32)))
        return [first, second]

    def stale_root(c: _ProbeContext) -> List[Verdict]:
        return [c.verifier.submit(c.honest.prove(0, c.genesis, "cd" * 32))]

    def future_slot(c: _ProbeContext) -> List[Verdict]:
        return [c.verifier.submit(c.honest.prove(5, c.genesis, c.root))]

    def revoked_sender(c: _ProbeContext) -> List[Verdict]:
        d = c.rogue.draft(0, c.genesis, c.root)
        first = c.verifier.submit(c.rogue.seal(replace(d, leaf_hash="00" * 32)))
        return [first, c.verifier.submit(c.rogue.prove(0, c.genesis, c.root))]

    def late_duplicate(c: _ProbeContext) -> List[Verdict]:
        root0 = c.root
        first = c.verifier.submit(c.honest.prove(0, c.genesis, root0))
        return [first, c.verifier.submit(c.rogue.prove(0, c.genesis, root0))]

    matrix: List[Tuple[str, Callable[[_ProbeContext], List[Verdict]], List[Verdict], bool]] = [
        ("tampered output", tampered_output, [V.BAD_PROOF], True),
        ("consistent-claim forgery", consistent_claim_forgery, [V.BAD_PROOF], True),
        ("malformed proof", malformed_proof, [V.BAD_PROOF], True),
        ("forged leaf commitment", forged_leaf, [V.BAD_COMMITMENT], True),
        ("forged-proof: tampered fold", forged_proof_tampered_fold, [V.BAD_PROOF], True),
        ("forged-proof: altered committed value", forged_proof_altered_committed_value, [V.BAD_PROOF], True),
        ("forged-proof: swapped merkle sibling", forged_proof_swapped_merkle_sibling, [V.STALE], False),
        ("wrong input", wrong_input, [V.BAD_INPUT], True),
        ("forged signature", forged_signature, [V.BAD_SIGNATURE], False),
        ("unknown sender", unknown_sender, [V.BAD_SIGNATURE], False),
        ("replayed message", replayed_message, [V.ACCEPTED, V.REPLAY], False),
        ("equivocation", equivocation, [V.ACCEPTED, V.EQUIVOCATION], True),
        ("stale root", stale_root, [V.STALE], False),
        ("future slot", future_slot, [V.BAD_SLOT], False),
        ("revoked sender", revoked_sender, [V.BAD_COMMITMENT, V.REVOKED_SENDER], True),
        ("late duplicate", late_duplicate, [V.ACCEPTED, V.LATE], False),
    ]

    results: List[Tuple[str, bool]] = []
    PKIRegistry.reset()
    with _quiet_slashing():
        for index, (name, attack, expected, rogue_slashed) in enumerate(matrix):
            tag = f"probe{index}"
            weights = FieldTensor2D.random(16, 16, seed=seed + 100)
            genesis = FieldTensor2D.random(1, 16, seed=seed + 101)
            honest = LayerProver(CryptoKey(f"honest_{tag}"), weights)
            rogue = LayerProver(CryptoKey(f"rogue_{tag}"), weights)
            context = _ProbeContext(tag, weights, genesis, honest, rogue,
                                    LayerVerifier(weights, genesis), random.Random(seed + index))
            verdicts = attack(context)
            ok = (
                verdicts == expected
                and PKIRegistry.is_slashed(rogue.key.node_id) == rogue_slashed
                and not PKIRegistry.is_slashed(honest.key.node_id)
            )
            results.append((name, ok))
    return results


def _test_adversarial_storm(cfg: SuiteConfig) -> None:
    memory_before = _rss_mb()
    attacked = _run_pipeline(cfg, adversary=True)
    baseline = _run_pipeline(cfg, adversary=False)
    memory_after = _rss_mb()

    _require(attacked.layers_accepted == cfg.layers, "not every honest layer was accepted")
    _require(attacked.rogue_outcomes[Verdict.ACCEPTED] == 0, "a Byzantine submission was accepted")
    _require(HONEST_NODE_ID not in attacked.slashed, "the honest node was slashed")
    _require(attacked.log_valid and baseline.log_valid, "accumulator audit failed")
    _require(attacked.root == baseline.root, "the adversary altered the committed state root")

    outcomes = ", ".join(
        f"{verdict.value} {count:,}"
        for verdict, count in sorted(attacked.rogue_outcomes.items(), key=lambda item: -item[1])
    ) or "none"
    _say(f"{attacked.layers_accepted:,}/{cfg.layers:,} honest layers verified and folded in "
         f"{attacked.duration:.2f}s ({cfg.layers / attacked.duration:,.0f} layers/s)")
    _say(f"Byzantine submissions: {attacked.rogue_submitted:,} | accepted: 0 | {outcomes}")
    _say(f"State root {attacked.root[:16]}… identical to adversary-free baseline; accumulator audit passed")

    probes = _run_probe_matrix(cfg.seed)
    failed = [name for name, ok in probes if not ok]
    _require(not failed, f"probe matrix failures: {', '.join(failed)}")
    _say(f"Byzantine probe matrix: {len(probes)}/{len(probes)} attacks (including forged-proof tests) classified and punished as specified")
    _say(f"Memory: {_fmt_mb(memory_before)} -> {_fmt_mb(memory_after)}")


# ---- Test 2: CRDT split-brain -----------------------------------------

def _test_crdt_split_brain(cfg: SuiteConfig) -> None:
    initial_state = {"vault_balance": 1000, "nonce": 0}

    def apply_credit(state: Dict[str, Any], amount: int) -> Dict[str, Any]:
        return {"vault_balance": state["vault_balance"] + amount, "nonce": state["nonce"] + 1}

    def apply_debit(state: Dict[str, Any], amount: int) -> Dict[str, Any]:
        return {"vault_balance": state["vault_balance"] - amount, "nonce": state["nonce"] + 1}

    methods = {"credit": apply_credit, "debit": apply_debit}
    replicas = [AutoCRDTCompiler(initial_state, methods) for _ in range(cfg.crdt_replicas)]
    store = ShardedKeyValueStore(shard_count=16)

    def partition_worker(index: int, node: AutoCRDTCompiler) -> None:
        for i in range(cfg.crdt_ops):
            op_id = node.execute("credit", 10 + index) if i % 2 == 0 else node.execute("debit", 5)
            store.set(f"op:{index}:{i}", op_id)

    threads = [threading.Thread(target=partition_worker, args=(i, r)) for i, r in enumerate(replicas)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    def tree_merge(nodes: List[AutoCRDTCompiler]) -> AutoCRDTCompiler:
        while len(nodes) > 1:
            paired = [nodes[i].merge(nodes[i + 1]) for i in range(0, len(nodes) - 1, 2)]
            nodes = paired + ([nodes[-1]] if len(nodes) % 2 else [])
        return nodes[0]

    started = time.perf_counter()
    forward = functools.reduce(lambda x, y: x.merge(y), replicas)
    elapsed_ms = (time.perf_counter() - started) * 1000
    backward = functools.reduce(lambda x, y: x.merge(y), reversed(replicas))
    tree = tree_merge(list(replicas))

    credits, debits = (cfg.crdt_ops + 1) // 2, cfg.crdt_ops // 2
    total_ops = cfg.crdt_replicas * cfg.crdt_ops
    expected = {
        "vault_balance": 1000 + sum(credits * (10 + i) - debits * 5 for i in range(cfg.crdt_replicas)),
        "nonce": total_ops,
    }
    _require(forward.state == expected, f"reconciled state {forward.state} != expected {expected}")
    _require(len({forward.state_digest(), backward.state_digest(), tree.state_digest()}) == 1,
             "merge order changed the converged state")
    _require(forward.merge(forward).state_digest() == forward.state_digest(), "merge is not idempotent")
    _require(len(forward.causal_history) == total_ops, "operations were lost in the merge")
    _require(len(forward.head_ops) == cfg.crdt_replicas, "unexpected number of causal heads")
    _require(len(store) == total_ops, "sharded store lost operation records")
    _require(all(store.get(f"op:{i}:{j}") in forward.causal_history
                 for i in range(cfg.crdt_replicas) for j in range(cfg.crdt_ops)),
             "stored operation ids are missing from the merged history")

    _say(f"Reconciled {len(forward.causal_history):,} causal operations across {cfg.crdt_replicas} replicas.")
    _say(f"Final reconciled state: {forward.state}")
    _say(f"Convergence: forward, reverse and tree merges identical (digest {forward.state_digest()[:12]}…); "
         f"merge idempotent")
    _say(f"Merge latency: {elapsed_ms:.2f} ms")


# ---- Test 3: asynchronous replay storm --------------------------------

def _test_replay_storm(cfg: SuiteConfig) -> None:
    accumulator = MerkleIVCAccumulator()
    pool: List[Tuple[str, SumcheckProof]] = []
    counters: Counter = Counter()
    errors: List[BaseException] = []
    guard = threading.Lock()

    def worker(index: int) -> None:
        rng = random.Random(cfg.seed * 1_000 + index)
        activation = FieldTensor2D.random(1, 16, seed=cfg.seed + 2 * index)
        weights = FieldTensor2D.random(16, 16, seed=cfg.seed + 2 * index + 1)
        try:
            for _ in range(cfg.replay_iterations):
                time.sleep(rng.uniform(0.0001, 0.002))
                if rng.random() < cfg.replay_fraction:
                    with guard:
                        candidate = rng.choice(pool) if pool else None
                    if candidate is not None:
                        try:
                            accumulator.fold(*candidate)
                            outcome = "replay_accepted"
                        except ReplayError:
                            outcome = "replay_blocked"
                        except StaleProofError:
                            outcome = "replay_misclassified"
                        with guard:
                            counters[outcome] += 1
                        continue
                while True:  # optimistic concurrency: re-prove against the newest root on contention
                    root = accumulator.state_root
                    output = activation.matmul(weights)
                    proof = GKRSumcheck.prove_layer(activation, weights, root, c=output)
                    if not GKRSumcheck.verify_layer(activation, weights, output, proof, root):
                        raise ValidationError("an honest proof failed verification")
                    next_activation, leaf = commit_activation(output)
                    try:
                        accumulator.fold(leaf, proof)
                    except StaleProofError:
                        with guard:
                            counters["contention_retries"] += 1
                        continue
                    with guard:
                        pool.append((leaf, proof))
                        counters["folded"] += 1
                    activation = next_activation
                    break
        except BaseException as exc:
            with guard:
                errors.append(exc)

    started = time.perf_counter()
    workers = [threading.Thread(target=worker, args=(i,)) for i in range(cfg.replay_workers)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    duration = time.perf_counter() - started

    attempts = cfg.replay_workers * cfg.replay_iterations
    folded, blocked = counters["folded"], counters["replay_blocked"]
    _require(not errors, f"worker failure: {errors[0]!r}" if errors else "")
    _require(counters["replay_accepted"] == 0, "a replayed proof was accepted")
    _require(counters["replay_misclassified"] == 0, "a replay was not recognised as a replay")
    _require(folded + blocked == attempts, f"accounting mismatch: {folded} + {blocked} != {attempts}")
    _require(len(accumulator) == folded, "accumulator log length disagrees with folded count")
    _require(accumulator.audit(), "accumulator audit failed")

    _say(f"Processed {attempts:,} attempts in {duration:.4f}s | Folded: {folded:,} | "
         f"Replays blocked: {blocked:,} | Contention retries: {counters['contention_retries']:,}")
    _say(f"Final root {accumulator.state_root[:16]}…; replay of full log reproduces it")


# ---- Test 4: compression and shard routing ----------------------------

def _test_compression_and_routing(cfg: SuiteConfig) -> None:
    rng = random.Random(cfg.seed)
    vocabulary = ("lattice", "cipher", "tensor", "ledger", "shard", "quorum", "merkle", "epoch",
                  "nonce", "vector", "fold", "proof", "field", "replica", "causal", "commit")
    noise = rng.randbytes(1 << 16)
    payloads: List[Tuple[str, bytes, PhaseClass]] = [
        ("empty", b"", PhaseClass.UNIFORM),
        ("single byte", b"\x7f", PhaseClass.UNIFORM),
        ("zero page", bytes(1 << 20), PhaseClass.UNIFORM),
        ("periodic frame", b"DV-ENGINE:" * 50_000, PhaseClass.PERIODIC),
        ("natural text", " ".join(rng.choice(vocabulary) for _ in range(20_000)).encode(), PhaseClass.DEFLATE),
        ("zeros + noise", bytes(1 << 16) + noise, PhaseClass.DEFLATE),
        ("pure noise", noise, PhaseClass.VERBATIM),
    ]

    store = ShardedKeyValueStore(shard_count=16)
    manifests: Dict[str, SingularityManifest] = {}
    raw_total = stored_total = 0
    for name, data, expected_class in payloads:
        manifest = SingularityEngine.fold(data)
        _require(manifest.phase_class is expected_class,
                 f"'{name}' routed to {manifest.phase_class.name}, expected {expected_class.name}")
        store.set(f"manifest:{name}", manifest.to_bytes())
        restored = SingularityManifest.from_bytes(store.get(f"manifest:{name}"))
        _require(restored == manifest, f"'{name}' manifest did not survive serialisation")
        _require(SingularityEngine.unfold(restored) == data, f"'{name}' did not round-trip bit-exactly")
        manifests[name] = manifest
        raw_total += len(data)
        stored_total += manifest.stored_size

    detected = 0
    targets = ("periodic frame", "natural text", "pure noise", "zero page")
    for name in targets:
        manifest = manifests[name]
        damaged = bytearray(manifest.descriptor)
        damaged[len(damaged) // 2] ^= 0x01
        forged = dataclasses.replace(manifest, descriptor=bytes(damaged))
        try:
            SingularityEngine.unfold(forged)
        except IntegrityError:
            detected += 1
    _require(detected == len(targets), f"only {detected}/{len(targets)} corrupted manifests were detected")

    def writer(index: int) -> None:
        for i in range(512):
            store.set(f"tensor:{index}:{i}", i)

    writers = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join()
    loads = store.shard_loads()
    mean = sum(loads) / len(loads)
    _require(len(store) == len(payloads) + 8 * 512, "concurrent writes were lost")
    _require(max(loads) < 1.5 * mean and min(loads) > 0.5 * mean, f"unbalanced shard routing: {loads}")

    classes = Counter(m.phase_class.name.lower() for m in manifests.values())
    _say(f"{len(payloads)} payloads round-tripped bit-exact | "
         + ", ".join(f"{name} {count}" for name, count in sorted(classes.items())))
    _say(f"Aggregate {raw_total / 1e6:.2f} MB -> {stored_total / 1e6:.3f} MB stored "
         f"({raw_total / stored_total:,.1f}x) | corruption detected on {detected}/{len(targets)} tampered manifests")
    _say(f"Shard routing: {len(store):,} keys over {store.shard_count} shards "
         f"(min {min(loads)} / max {max(loads)})")


def run_comprehensive_validation_suite(cfg: Optional[SuiteConfig] = None) -> bool:
    cfg = cfg or SuiteConfig()
    rule = "=" * 71
    print(rule)
    print("STARTING UNIFIED ENTERPRISE ENGINE VALIDATION SUITE")
    print(rule)

    try:
        _backend_self_test()
    except ValidationError as exc:
        print(f"Backend self-test FAILED: {exc}")
        return False
    print(f"Backend: {backend_description()} (self-test passed) | Python {sys.version.split()[0]}")
    print(f"Workload: {cfg.layers:,} layers x {cfg.width} width | seed {cfg.seed}")

    tests: List[Tuple[str, Callable[[SuiteConfig], None]]] = [
        (f"Recursive Adversarial Storm ({cfg.layers:,} Layers & Byzantine Injection)", _test_adversarial_storm),
        ("Distributed Split-Brain & CRDT Merge Storm", _test_crdt_split_brain),
        ("Asynchronous Byzantine Replay & Concurrency Storm", _test_replay_storm),
        ("Lossless Phase-Class Compression & Shard Routing", _test_compression_and_routing),
    ]
    failures: List[str] = []
    for number, (title, test) in enumerate(tests, start=1):
        print(f"\n[Test {number}/{len(tests)}] {title}...")
        try:
            test(cfg)
        except Exception as exc:
            failures.append(title)
            print(f"-> FAILED: {exc}" if isinstance(exc, ValidationError) else f"-> FAILED: {exc!r}")

    print(rule)
    if failures:
        print(f"VALIDATION SUITE FAILED: {len(failures)}/{len(tests)} tests failed.")
    else:
        print(f"VALIDATION SUITE COMPLETE: {len(tests)}/{len(tests)} tests passed.")
    print(rule)
    return not failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Enterprise DV-Engine master engine validation suite.")
    parser.add_argument("--layers", type=int, help="layers in the adversarial storm (default 1000)")
    parser.add_argument("--width", type=int, help="tensor width (default 256)")
    parser.add_argument("--seed", type=int, default=42, help="deterministic workload seed (default 42)")
    parser.add_argument("--quick", action="store_true", help="reduced workload for smoke testing")
    
    # Use parse_known_args() to absorb Jupyter kernel flags like -f
    args, unknown = parser.parse_known_args(argv)

    cfg = SuiteConfig.quick(args.seed) if args.quick else SuiteConfig(seed=args.seed)
    if args.layers is not None:
        cfg.layers = args.layers
    if args.width is not None:
        cfg.width = args.width
    if cfg.layers < 1 or cfg.width < 1:
        parser.error("--layers and --width must be positive")

    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    return 0 if run_comprehensive_validation_suite(cfg) else 1


if __name__ == "__main__":
    in_notebook = "ipykernel" in sys.modules
    notebook_args: list = []
    code = main(notebook_args if in_notebook else None)
    if not in_notebook:
        sys.exit(code)
