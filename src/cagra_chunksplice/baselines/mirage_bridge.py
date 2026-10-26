"""ctypes binding to the CPU upstream C++ bridge built by setup_external.py."""
from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class MirageBridge:
    def __init__(self, library, dimension, s=32, r=4, iterations=15):
        self.library_path = str(Path(library).resolve(strict=True))
        self.lib = ctypes.CDLL(self.library_path)
        self.handle = None
        self.dimension = dimension
        self.mirage = SimpleNamespace(S=s, R=r, iter=iterations)
        specs = {
            "cs_mirage_create": ([ctypes.c_int] * 4, ctypes.c_void_p),
            "cs_mirage_destroy": ([ctypes.c_void_p], None),
            "cs_mirage_add": ([ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p], ctypes.c_int),
            "cs_mirage_search": ([ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                  ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
            "cs_mirage_ntotal": ([ctypes.c_void_p], ctypes.c_int64),
            "cs_mirage_error": ([], ctypes.c_char_p),
            "cs_mirage_version": ([], ctypes.c_char_p),
            "cs_mirage_threads": ([], ctypes.c_int),
        }
        for name, (args, result) in specs.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.version = self.lib.cs_mirage_version().decode()
        self.threads = self.lib.cs_mirage_threads()
        # Bridges built before the patch series do not export cs_mirage_patches.
        self.patches = None
        if hasattr(self.lib, "cs_mirage_patches"):
            self.lib.cs_mirage_patches.argtypes, self.lib.cs_mirage_patches.restype = [], ctypes.c_char_p
            self.patches = [entry for entry in self.lib.cs_mirage_patches().decode().split(",") if entry]
        self.handle = self.lib.cs_mirage_create(dimension, s, r, iterations)
        if not self.handle:
            raise RuntimeError(self.lib.cs_mirage_error().decode())

    def _check(self, code):
        if code != 0:
            raise RuntimeError("Upstream MIRAGE: " + self.lib.cs_mirage_error().decode())

    def _matrix(self, value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != self.dimension or not np.isfinite(value).all():
            raise ValueError("invalid native MIRAGE matrix")
        if not self.handle:
            raise RuntimeError("MIRAGE bridge is closed")
        return value

    def add(self, vectors):
        vectors = self._matrix(vectors)
        before = self.lib.cs_mirage_ntotal(self.handle)
        self._check(self.lib.cs_mirage_add(self.handle, len(vectors), vectors.ctypes.data))
        if self.lib.cs_mirage_ntotal(self.handle) != before + len(vectors):
            raise RuntimeError("native MIRAGE did not append the requested vectors")

    def search(self, queries, k):
        queries = self._matrix(queries)
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        distances = np.empty((len(queries), k), dtype=np.float32)
        ids = np.empty((len(queries), k), dtype=np.int64)
        self._check(self.lib.cs_mirage_search(self.handle, len(queries), queries.ctypes.data,
                                             k, distances.ctypes.data, ids.ctypes.data))
        return distances, ids

    def close(self):
        if self.handle:
            self.lib.cs_mirage_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
