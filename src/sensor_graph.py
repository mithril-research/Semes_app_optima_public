"""Utilities for using the process sensor graph in scoring and RCA.

The graph file may contain pseudo nodes such as ``input`` and ``out1``. Those
nodes are kept for path/rank computation but are not treated as data columns.
Range names such as ``h_0211-4_PV`` and ``9000015-9000018`` are expanded.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


_RANGE_RE = re.compile(r"^(?P<prefix>.*?)(?P<start>\d+)-(?P<end>\d+)(?P<suffix>[^0-9]*)$")
_PSEUDO_RE = re.compile(r"^(input|out\d*)$", re.IGNORECASE)
_GRAPH_CACHE = {}


def is_pseudo_node(name: str) -> bool:
    return bool(_PSEUDO_RE.match(str(name).strip()))


def expand_node_name(name: str) -> list[str]:
    """Expand compact graph names into concrete sensor/IO names."""
    name = str(name).strip()
    m = _RANGE_RE.match(name)
    if not m:
        return [name]
    prefix = m.group("prefix")
    start_s = m.group("start")
    end_s = m.group("end")
    suffix = m.group("suffix")

    if len(end_s) < len(start_s):
        end_s = start_s[:len(start_s) - len(end_s)] + end_s
    start = int(start_s)
    end = int(end_s)
    if end < start:
        return [name]

    pad = len(start_s) if len(start_s) > 1 else 0
    out = []
    for value in range(start, end + 1):
        num = str(value).zfill(pad) if pad else str(value)
        out.append(f"{prefix}{num}{suffix}")
    return out


class SensorGraph:
    def __init__(self, edges: dict[str, set[str]]):
        self.edges = {str(k): set(map(str, v)) for k, v in edges.items()}
        self.rev = defaultdict(set)
        self.nodes = set(self.edges)
        for src, dsts in self.edges.items():
            for dst in dsts:
                self.nodes.add(dst)
                self.rev[dst].add(src)
        self.rank = self._build_rank()

    @classmethod
    def from_json(cls, path: Path) -> "SensorGraph":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        edges = defaultdict(set)
        for src, dsts in raw.items():
            srcs = expand_node_name(src)
            expanded_dsts = []
            for dst in dsts:
                expanded_dsts.extend(expand_node_name(dst))
            for s in srcs:
                edges[s].update(expanded_dsts)
        return cls(edges)

    @classmethod
    def empty(cls) -> "SensorGraph":
        return cls({})

    def _build_rank(self) -> dict[str, int]:
        if not self.nodes:
            return {}
        starts = ["input"] if "input" in self.nodes else []
        if not starts:
            starts = sorted(n for n in self.nodes if not self.rev.get(n))
        q = deque(starts)
        seen = set(starts)
        order = []
        while q:
            cur = q.popleft()
            order.append(cur)
            for nxt in sorted(self.edges.get(cur, ())):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        for node in sorted(self.nodes - seen):
            order.append(node)
        return {node: i for i, node in enumerate(order)}

    def sensor_rank(self, sensor_cols) -> dict[str, int]:
        base = len(self.rank)
        return {str(name): self.rank.get(str(name), base + i)
                for i, name in enumerate(sensor_cols)}

    def measured_neighbors(self, sensor_cols, max_hops: int = 2) -> dict[int, list[int]]:
        """Return measured sensor neighbors reachable through the graph.

        The BFS is undirected for scoring support: an anomaly is more plausible
        when graph-adjacent measured sensors also show residual energy.
        """
        names = [str(x) for x in sensor_cols]
        idx = {name: i for i, name in enumerate(names)}
        out = {}
        max_hops = max(1, int(max_hops or 1))
        for name, i in idx.items():
            q = deque([(name, 0)])
            seen = {name}
            hits = []
            while q:
                cur, depth = q.popleft()
                if depth >= max_hops:
                    continue
                nxts = set(self.edges.get(cur, ())) | set(self.rev.get(cur, ()))
                for nxt in sorted(nxts, key=lambda x: self.rank.get(x, 10**9)):
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    if nxt in idx and nxt != name:
                        hits.append(idx[nxt])
                    q.append((nxt, depth + 1))
            out[i] = sorted(set(hits))
        return out

    def adjust_abs_z(self, z_abs, sensor_cols, alpha: float = 0.25,
                     threshold: float = 4.0, max_hops: int = 2):
        """Boost residual energy when graph-neighbor sensors are also active."""
        if alpha <= 0 or not self.nodes:
            return np.asarray(z_abs, dtype=float)
        arr = np.asarray(z_abs, dtype=float)
        if arr.size == 0 or arr.shape[-1] != len(sensor_cols):
            return arr
        neighbors = self.measured_neighbors(sensor_cols, max_hops=max_hops)
        if not any(neighbors.values()):
            return arr

        flat = arr.reshape(-1, arr.shape[-1])
        corrected = flat.copy()
        ref = max(float(threshold or 0.0), 1e-12)
        for j, neigh in neighbors.items():
            if not neigh:
                continue
            support = np.nanmax(flat[:, neigh], axis=1)
            support = np.nan_to_num(support, nan=0.0, posinf=ref, neginf=0.0)
            boost = float(alpha) * np.clip(support / ref, 0.0, 1.0)
            corrected[:, j] = flat[:, j] * (1.0 + boost)
        return corrected.reshape(arr.shape)


def _graph_path_from_cfg(cfg) -> Path:
    path = getattr(cfg, "sensors_order_path", None)
    if path:
        return Path(path)
    root = Path(getattr(cfg, "artifacts_dir", ".")).resolve().parent
    return root / "sensors_order.json"


def load_sensor_graph(cfg=None, path: Path | None = None) -> SensorGraph:
    graph_path = Path(path) if path is not None else _graph_path_from_cfg(cfg)
    key = str(graph_path.resolve()) if graph_path.exists() else str(graph_path)
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached
    if not graph_path.exists():
        graph = SensorGraph.empty()
    else:
        try:
            graph = SensorGraph.from_json(graph_path)
        except Exception:
            graph = SensorGraph.empty()
    _GRAPH_CACHE[key] = graph
    return graph


def graph_sensor_rank(sensor_cols, cfg) -> dict[str, int]:
    graph = load_sensor_graph(cfg)
    if not graph.nodes:
        preferred = [str(x) for x in getattr(cfg, "sensor_causal_order", ()) or ()]
        rank = {name: i for i, name in enumerate(preferred)}
        base = len(rank)
        return {str(name): rank.get(str(name), base + i)
                for i, name in enumerate(sensor_cols)}
    return graph.sensor_rank(sensor_cols)


def graph_adjust_abs_z(z_abs, sensor_cols, cfg):
    if not bool(getattr(cfg, "graph_score_enabled", True)):
        return np.asarray(z_abs, dtype=float)
    graph = load_sensor_graph(cfg)
    return graph.adjust_abs_z(
        z_abs,
        sensor_cols,
        alpha=float(getattr(cfg, "graph_score_alpha", 0.25)),
        threshold=float(getattr(cfg, "flag_sigma", 4.0)),
        max_hops=int(getattr(cfg, "graph_score_max_hops", 2)),
    )

