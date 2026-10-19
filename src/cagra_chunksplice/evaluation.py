"""Common query evaluation for all experimental methods."""
from __future__ import annotations

from cagra_chunksplice.search import graph_search
from cagra_chunksplice.timing import measured
from cagra_chunksplice.metrics import recall
from cagra_chunksplice.timing import sample_summary


def evaluate_graph(x, graph, queries, truth, latest, latest_truth, entries, old_count, a):
    pred, query_ms, samples = measured(
        lambda: graph_search(x, graph, queries, entries, a), x.device, a.repeats)
    latest_pred, latest_ms, latest_samples = measured(
        lambda: graph_search(x, graph, latest, entries, a), x.device, a.repeats)
    return {
        "query_ms": query_ms, "query_samples_ms": samples,
        "query_sample_summary": sample_summary(samples),
        "qps": len(queries) * 1000.0 / query_ms,
        "recall10": recall(pred, truth),
        "old_query_recall10": recall(pred[:old_count], truth[:old_count]),
        "new_query_recall10": recall(pred[old_count:], truth[old_count:]),
        "latest_chunk_query_recall10": recall(latest_pred, latest_truth),
        "latest_chunk_query_ms": latest_ms, "latest_chunk_query_samples_ms": latest_samples,
        "latest_chunk_query_sample_summary": sample_summary(latest_samples),
    }
