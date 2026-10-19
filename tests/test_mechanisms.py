"""Mechanism checks on CPU and, when available, CUDA."""
from types import SimpleNamespace
import unittest

import torch

from cagra_chunksplice.algorithms.chunk_splice import chunk_splice_extend
from cagra_chunksplice.algorithms.naive import naive_extend
from cagra_chunksplice.core import exact_knn, squared_l2, unique_candidates
from cagra_chunksplice.graph import build_graph, validate_graph
from cagra_chunksplice.metrics import intra_edges
from cagra_chunksplice.selftest import self_test


class MechanismTests(unittest.TestCase):
    def test_original_mechanism_checks_cpu(self):
        with torch.inference_mode():
            self_test(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_original_mechanism_checks_cuda(self):
        with torch.inference_mode():
            self_test(torch.device("cuda"))

    def test_blocked_knn_and_deduplication(self):
        x = torch.tensor([[0., 0.], [1., 0.], [3., 0.], [7., 0.]])
        truth = squared_l2(x, x)
        truth.fill_diagonal_(float("inf"))
        for block in (1, 3, 8):
            actual = exact_knn(x, x, 2, block, torch.arange(len(x)))
            self.assertTrue(torch.equal(actual, truth.topk(2, largest=False).indices))
        ids = torch.tensor([[2, 2, -1, 0, 3]])
        unique = unique_candidates(ids, torch.tensor([0]))
        self.assertEqual(sorted(unique[unique >= 0].tolist()), [2, 3])

    def test_splice_budget_and_immutable_predecessor_across_updates(self):
        args = SimpleNamespace(degree=4, distance_batch=8, splice_intra=2,
                               splice_pool=6, splice_alpha=1.0)
        generator = torch.Generator().manual_seed(19)
        x = torch.randn(19, 3, generator=generator)
        graph = build_graph(x[:12], args)
        with torch.inference_mode():
            # Include a singleton and a final remainder chunk.
            for start, end in ((12, 16), (16, 17), (17, 19)):
                before = graph.clone()
                naive = naive_extend(x[:end], graph, start, end, args)
                spliced = chunk_splice_extend(x[:end], graph, start, end, args)
                self.assertTrue(torch.equal(graph, before))
                self.assertTrue(torch.equal(spliced[:start], naive[:start]))
                validate_graph(spliced, end)
                if end - start == 1:
                    self.assertTrue(torch.equal(spliced, naive))
                else:
                    bridges = ((spliced[start:end] >= 0) & (spliced[start:end] < start)).sum(1)
                    reserve = args.degree - min(args.splice_intra, end - start - 1)
                    self.assertTrue(bool((bridges >= reserve).all()))
                    self.assertGreater(intra_edges(spliced, start, end - start), 0)
                graph = spliced

    def test_graph_validation_rejects_invalid_adjacency(self):
        for graph in (torch.tensor([[0], [0]]), torch.tensor([[2], [0]]),
                      torch.tensor([[-2], [0]]), torch.tensor([[1, 1], [0, -1]])):
            with self.subTest(graph=graph.tolist()), self.assertRaises(AssertionError):
                validate_graph(graph, 2)
