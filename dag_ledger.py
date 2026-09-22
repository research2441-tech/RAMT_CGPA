from __future__ import annotations
import hashlib, json, time, threading, concurrent.futures
from dataclasses import dataclass, asdict
from typing import List, Tuple

def h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

@dataclass
class Transaction:
    tx_id: str
    parents: Tuple[str,str]
    timestamp_ns: int
    payload_hash: str
    stage: int
    cumulative_weight: int = 1

class DAGLedger:
    def __init__(self):
        genesis = Transaction("GENESIS", ("",""), time.time_ns(), h(b"GENESIS"), 0, 1)
        self.txs = {"GENESIS": genesis}
        self.tips = ["GENESIS"]
        self.lock = threading.Lock()

    def _select_parents(self):
        tips = list(self.tips[-16:]) or ["GENESIS"]
        if len(tips) == 1: return (tips[0], tips[0])
        return (tips[-1], tips[-2])

    def submit(self, payload: bytes, stage=0):
        with self.lock:
            parents = self._select_parents()
            stamp = time.time_ns()
            pid = h(payload)
            txid = h(f"{parents}|{stamp}|{pid}|{stage}".encode())
            tx = Transaction(txid, parents, stamp, pid, int(stage), 1)
            self.txs[txid] = tx
            self.tips.append(txid)
            # Bound tip list to avoid quadratic state.
            self.tips = self.tips[-128:]
            for p in set(parents):
                if p in self.txs:
                    self.txs[p].cumulative_weight += 1
            return tx

    def verify(self, tx: Transaction, payload: bytes):
        return tx.payload_hash == h(payload) and all((p in self.txs or p=="") for p in tx.parents)

    def estimated_bytes(self):
        return sum(len(json.dumps(asdict(t), separators=(",",":"))) for t in self.txs.values())

def benchmark_dag(n_transactions=1000, workers=8, payload_bytes=96):
    ledger = DAGLedger()
    payload = b"x"*payload_bytes
    lat = []
    t0 = time.perf_counter()
    def one(i):
        a = time.perf_counter_ns()
        tx = ledger.submit(payload + i.to_bytes(8,"big"), stage=i%5)
        ok = ledger.verify(tx, payload + i.to_bytes(8,"big"))
        return (time.perf_counter_ns()-a)/1e6, ok
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for ms, ok in ex.map(one, range(n_transactions)):
            lat.append(ms)
            if not ok: raise RuntimeError("DAG verification failed")
    elapsed = max(time.perf_counter()-t0, 1e-9)
    return {
        "transactions": n_transactions,
        "workers": workers,
        "throughput_tps": n_transactions/elapsed,
        "latency_mean_ms": sum(lat)/len(lat),
        "latency_p95_ms": sorted(lat)[int(.95*(len(lat)-1))],
        "ledger_bytes": ledger.estimated_bytes(),
    }
