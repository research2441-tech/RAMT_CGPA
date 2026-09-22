from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass
from crypto_governance import split_secret, reconstruct_secret
from dag_ledger import DAGLedger

@dataclass
class Proof:
    statement_hash: str
    transcript_commitment: str
    severity: int
    confidence_milli: int

class ResearchProofEngine:
    """
    Deterministic research proof interface for end-to-end timing.
    It reproduces the control-flow of a verifiable state transition.
    It does not claim to be a production zk-STARK implementation.
    """
    def prove(self, sealed_input: bytes, severity: int, confidence: float):
        st = hashlib.sha256(sealed_input).hexdigest()
        cm = int(round(confidence*1000))
        transcript = hashlib.sha256(f"{st}|{severity}|{cm}|RAMT-CGPA".encode()).hexdigest()
        return Proof(st, transcript, int(severity), cm)

    def verify(self, proof: Proof, sealed_input: bytes):
        st = hashlib.sha256(sealed_input).hexdigest()
        exp = hashlib.sha256(f"{st}|{proof.severity}|{proof.confidence_milli}|RAMT-CGPA".encode()).hexdigest()
        return st == proof.statement_hash and exp == proof.transcript_commitment

def benchmark_breakglass(secret=b"A"*32, n=7, normal_k=5, emergency_k=2,
                         severity=4, confidence=.97, ledger=None):
    ledger = ledger or DAGLedger()
    shares = split_secret(secret, n=n, k=normal_k, seed=42)
    engine = ResearchProofEngine()
    sealed = b"cardiac-state-digest"

    t0 = time.perf_counter_ns()
    p0 = time.perf_counter_ns()
    proof = engine.prove(sealed, severity, confidence)
    prove_ms = (time.perf_counter_ns()-p0)/1e6

    v0 = time.perf_counter_ns()
    valid = engine.verify(proof, sealed)
    verify_ms = (time.perf_counter_ns()-v0)/1e6
    if not valid: raise RuntimeError("Proof verification failed")

    l0 = time.perf_counter_ns()
    tx = ledger.submit(proof.transcript_commitment.encode(), stage=severity)
    ledger_ms = (time.perf_counter_ns()-l0)/1e6

    # For reproducibility, threshold transition is represented by reconstruction
    # from a fresh emergency-threshold share set of the same protected secret.
    r0 = time.perf_counter_ns()
    emergency_shares = split_secret(secret, n=n, k=emergency_k, seed=43)
    recovered = reconstruct_secret(emergency_shares[:emergency_k], length=len(secret))
    reconstruct_ms = (time.perf_counter_ns()-r0)/1e6

    total_ms = (time.perf_counter_ns()-t0)/1e6
    return {
        "prove_ms": prove_ms,
        "verify_ms": verify_ms,
        "ledger_ms": ledger_ms,
        "reconstruction_ms": reconstruct_ms,
        "total_breakglass_ms": total_ms,
        "reconstruction_ok": recovered == secret,
        "tx_id": tx.tx_id,
    }
