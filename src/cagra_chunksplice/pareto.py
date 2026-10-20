"""Empirical recall/throughput frontiers; never interpolate missing recall."""
from __future__ import annotations

import math
from itertools import product


def parameter_grid(a):
    grid = [(name, {"ef": ef}) for name, ef in
            product(("Naive", "ChunkSplice"), a.graph_ef_values)]
    grid += [("FAISS-HNSW", {"hnsw_m": m, "ef": ef})
             for m, ef in product(a.hnsw_m_values, a.hnsw_ef_search_values)]
    grid += [("MIRAGE", {"mirage_s": s, "mirage_r": r, "mirage_iter": it})
             for s, r, it in product(a.mirage_s_values, a.mirage_r_values, a.mirage_iter_values)]
    grid += [("cuVS-CAGRA", {"ef": ef, "search_width": width, "steps": steps})
             for ef, width, steps in product(a.cagra_itopk_values,
                                           a.cagra_search_width_values,
                                           a.cagra_iterations_values)]
    return grid


def frontier(points):
    """Maximize measured recall and QPS within one implementation group."""
    if any(not math.isfinite(p["recall10"]) or not 0 <= p["recall10"] <= 1
           or not math.isfinite(p["qps"]) or p["qps"] <= 0 for p in points):
        raise ValueError("Pareto points require finite recall and positive QPS")
    kept = []
    for point in points:
        if not any(other["recall10"] >= point["recall10"] and other["qps"] >= point["qps"]
                   and (other["recall10"] > point["recall10"] or other["qps"] > point["qps"])
                   for other in points):
            kept.append(point)
    return sorted(kept, key=lambda p: (p["recall10"], -p["qps"]))


def summarize_curves(points, targets):
    groups = {}
    for point in points:
        # Native and simulated samples must never share a frontier.
        key = point["method"] + (" / simulated" if point["simulated"] else " / native")
        groups.setdefault(key, []).append(point)
    curves = {key: frontier(rows) for key, rows in groups.items()}
    matched = {}
    for target in targets:
        matched[str(target)] = {}
        for key, rows in curves.items():
            eligible = [row for row in rows if row["recall10"] >= target]
            matched[str(target)][key] = max(eligible, key=lambda p: p["qps"]) if eligible else None
    return {"frontiers": curves, "matched_recall": matched,
            "matching_rule": "fastest measured point with recall >= target; null if unattained; no interpolation",
            "objectives": ["maximize recall10", "maximize qps"],
            "update_latency": "retained per point; not an axis of this search frontier"}


def curves_svg(curves):
    """Dependency-free artifact from measured points only."""
    from html import escape
    colors = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00")
    groups = curves["frontiers"]
    max_qps = max((p["qps"] for rows in groups.values() for p in rows), default=1)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="600" viewBox="0 0 960 600">',
             '<rect width="960" height="600" fill="white"/>',
             '<text x="60" y="30" font-size="18">Measured recall / throughput Pareto frontiers</text>',
             '<path d="M60 60V500H660" fill="none" stroke="black"/>',
             '<text x="300" y="550">Recall@10 (0 to 1)</text>',
             '<text x="65" y="55">QPS (0 to ' + str(max_qps) + ')</text>']
    for i, (name, rows) in enumerate(groups.items()):
        color = colors[i % len(colors)]
        coords = [(60 + p["recall10"] * 600, 500 - p["qps"] / max_qps * 430) for p in rows]
        line = " ".join(f"{x},{y}" for x, y in coords)
        parts.append(f'<polyline points="{line}" fill="none" stroke="{color}"/>')
        for x, y in coords:
            parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
        parts.append(f'<text x="680" y="{80+i*30}" fill="{color}">{escape(name)}</text>')
    parts.append('<text x="60" y="585">Lines connect observed frontier points; intermediate performance is unmeasured.</text></svg>')
    return "\n".join(parts)
