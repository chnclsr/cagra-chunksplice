// C ABI for the upstream CPU IndexMirage plus the patches in patches/series.
#if __has_include(<faiss/IndexMIRAGE.h>)
#include <faiss/IndexMIRAGE.h>
#else
#include <faiss/IndexMirage.h>
#endif
#include <omp.h>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#ifndef MIRAGE_PATCHES
#define MIRAGE_PATCHES ""
#endif

namespace {
thread_local std::string last_error;
template<class F> int guarded(F operation) noexcept {
    try { last_error.clear(); operation(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); }
    catch (...) { last_error = "Unknown upstream MIRAGE exception"; }
    return -1;
}
}
extern "C" {
const char* cs_mirage_error() { return last_error.c_str(); }
const char* cs_mirage_version() { return MIRAGE_REVISION; }
const char* cs_mirage_patches() { return MIRAGE_PATCHES; }
int cs_mirage_threads() { return omp_get_max_threads(); }
void* cs_mirage_create(int dimension, int s, int r, int iterations) noexcept {
    void* result = nullptr;
    guarded([&] {
        auto index = std::make_unique<faiss::IndexMirage>(dimension);
        index->mirage.S = s;
        index->mirage.R = r;
        index->mirage.iter = iterations;
        result = index.release();
    });
    return result;
}
int cs_mirage_add(void* ptr, std::int64_t n, const float* data) noexcept {
    return guarded([&] {
        auto* index = static_cast<faiss::IndexMirage*>(ptr);
        const std::int64_t before = index->ntotal;
        index->add(n, data);
        // Unpatched upstream double-inserts incremental batches, so ntotal != before + n.
        if (index->ntotal != before + n) {
            throw std::runtime_error(
                "IndexMirage::add appended " + std::to_string(index->ntotal - before) +
                " rows for " + std::to_string(n) + " vectors; apply "
                "native/mirage_bridge/patches (double-insert fix) and rebuild");
        }
    });
}
int cs_mirage_search(void* ptr, std::int64_t n, const float* q, std::int64_t k,
                     float* distances, std::int64_t* ids) noexcept {
    return guarded([&] { static_cast<faiss::IndexMirage*>(ptr)->search(n, q, k, distances, ids); });
}
std::int64_t cs_mirage_ntotal(void* ptr) {
    return static_cast<faiss::IndexMirage*>(ptr)->ntotal;
}
void cs_mirage_destroy(void* ptr) noexcept {
    delete static_cast<faiss::IndexMirage*>(ptr);
}
}
