from __future__ import annotations
import hashlib, hmac, json, os, secrets, struct, time
from dataclasses import dataclass
import numpy as np

try:
    from blake3 import blake3 as _b3
except Exception:
    _b3 = None

def digest(data: bytes) -> bytes:
    return _b3(data).digest() if _b3 else hashlib.blake2b(data, digest_size=32).digest()

def merkle_root(chunks):
    nodes = [digest(c) for c in chunks] or [digest(b"")]
    while len(nodes) > 1:
        if len(nodes) % 2: nodes.append(nodes[-1])
        nodes = [digest(nodes[i]+nodes[i+1]) for i in range(0,len(nodes),2)]
    return nodes[0]

def gaussian_ldp(x, epsilon=1.0, delta=1e-5, sensitivity=1.0, seed=42):
    sigma = sensitivity * np.sqrt(2*np.log(1.25/delta)) / max(epsilon, 1e-12)
    rng = np.random.default_rng(seed)
    return np.asarray(x, float) + rng.normal(0.0, sigma, size=np.asarray(x).shape), float(sigma)

# Prime comfortably larger than 2^256 for compact Shamir-style secret sharing.
P = 2**521 - 1

def _inv(a,p=P): return pow(a, p-2, p)

def split_secret(secret: bytes, n=7, k=5, seed=42):
    if not (2 <= k <= n): raise ValueError("Require 2 <= threshold <= shares")
    s = int.from_bytes(secret, "big")
    if s >= P: raise ValueError("Secret too large for field")
    rng = np.random.default_rng(seed)
    coeffs = [s] + [int.from_bytes(rng.bytes(64), "big") % P for _ in range(k-1)]
    shares = []
    for x in range(1,n+1):
        y = sum(c*pow(x,i,P) for i,c in enumerate(coeffs)) % P
        shares.append((x,y))
    return shares

def reconstruct_secret(shares, length=32):
    total = 0
    for i,(xi,yi) in enumerate(shares):
        num, den = 1,1
        for j,(xj,_) in enumerate(shares):
            if i == j: continue
            num = (num * (-xj)) % P
            den = (den * (xi-xj)) % P
        total = (total + yi*num*_inv(den)) % P
    return int(total).to_bytes(length, "big")

class KEMWrapper:
    """
    Attempts a liboqs KEM. Falls back to deterministic benchmark-only encapsulation.
    The fallback is not a post-quantum cryptographic substitute.
    """
    def __init__(self, algorithm="Kyber1024"):
        self.algorithm = algorithm
        self.backend = "benchmark_fallback"
        self._oqs = None
        try:
            import oqs
            candidates = [algorithm, "ML-KEM-1024", "Kyber1024"]
            enabled = set(oqs.get_enabled_kem_mechanisms())
            chosen = next((x for x in candidates if x in enabled), None)
            if chosen:
                self._oqs = oqs
                self.algorithm = chosen
                self.backend = "liboqs"
        except Exception:
            pass

    def keygen(self):
        if self._oqs:
            kem = self._oqs.KeyEncapsulation(self.algorithm)
            pk = kem.generate_keypair()
            sk = kem.export_secret_key()
            return pk, sk
        sk = secrets.token_bytes(32)
        pk = digest(b"PUB"+sk)
        return pk, sk

    def encapsulate(self, pk):
        if self._oqs:
            kem = self._oqs.KeyEncapsulation(self.algorithm)
            ct, ss = kem.encap_secret(pk)
            return ct, ss
        nonce = secrets.token_bytes(32)
        ct = nonce + digest(pk+nonce)
        ss = digest(pk+nonce+b"SS")
        return ct, ss

    def decapsulate(self, sk, ct):
        if self._oqs:
            kem = self._oqs.KeyEncapsulation(self.algorithm, sk)
            return kem.decap_secret(ct)
        pk = digest(b"PUB"+sk)
        nonce = ct[:32]
        return digest(pk+nonce+b"SS")

def protect_payload(payload: bytes, algorithm="Kyber1024"):
    kem = KEMWrapper(algorithm)
    pk, sk = kem.keygen()
    ct, ss = kem.encapsulate(pk)
    # Stream-XOR for research transport envelope only.
    stream = b""
    counter = 0
    while len(stream) < len(payload):
        stream += digest(ss + counter.to_bytes(8,"big"))
        counter += 1
    cipher = bytes(a^b for a,b in zip(payload,stream))
    tag = hmac.new(ss, cipher, hashlib.sha256).digest()
    return {
        "backend": kem.backend, "algorithm": kem.algorithm,
        "public_key": pk, "secret_key": sk, "kem_ciphertext": ct,
        "payload_ciphertext": cipher, "tag": tag
    }

def timed(fn, *args, **kwargs):
    t0 = time.perf_counter_ns()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter_ns()-t0)/1e6
