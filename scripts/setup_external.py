#!/usr/bin/env python3
"""Probe Linux/WSL2, build CPU FAISS/MIRAGE, and install native cuVS.

No system packages or Windows features are changed. Run inside an activated
Linux Python environment; --probe is read-only and --dry-run prints commands.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def capture(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        return {"ok": result.returncode == 0, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(), "returncode": result.returncode}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}


def detect_environment():
    kernel = platform.release().lower()
    wsl = "microsoft" in kernel or bool(os.environ.get("WSL_INTEROP"))
    wsl2 = wsl and ("wsl2" in kernel or "microsoft-standard" in kernel)
    tools = {name: shutil.which(name) for name in ("git", "cmake", "c++", "swig", "nvcc", "nvidia-smi")}
    # WSL may omit nvidia-smi from PATH.
    if not tools["nvidia-smi"] and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        tools["nvidia-smi"] = "/usr/lib/wsl/lib/nvidia-smi"
    nvidia = capture([tools["nvidia-smi"], "--query-gpu=name,driver_version,memory.total",
                      "--format=csv,noheader"]) if tools["nvidia-smi"] else {"ok": False}
    nvcc = capture([tools["nvcc"], "--version"]) if tools["nvcc"] else {"ok": False}
    match = re.search(r"release (\d+)\.", nvcc.get("stdout", ""))
    supported = platform.system() == "Linux" and (not wsl or wsl2)
    return {"system": platform.system(), "kernel": kernel, "machine": platform.machine(),
            "python": sys.version, "executable": sys.executable,
            "wsl": wsl, "wsl2": wsl2, "native_platform_supported": supported,
            "tools": tools, "gpu": nvidia, "nvcc": nvcc,
            "cuda_toolkit_major": int(match.group(1)) if match else None,
            "guidance": "Use native Linux or Ubuntu on WSL2 with a host NVIDIA driver. "
                        "WSL1 and Windows Python cannot run this native suite."}


def run(command, *, cwd=None, dry_run=False):
    print((str(cwd) + ": " if cwd else "") + shlex.join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def checkout(url, ref, destination, dry_run):
    if not destination.exists():
        run(["git", "clone", "--no-checkout", url, destination], dry_run=dry_run)
        run(["git", "-C", destination, "checkout", "--detach", ref], dry_run=dry_run)
    elif not dry_run:
        # Never reset a user's source tree or silently change its selected revision.
        origin = subprocess.check_output(["git", "-C", str(destination), "remote", "get-url", "origin"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain"], text=True).strip()
        current = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        wanted = subprocess.check_output(["git", "-C", str(destination), "rev-parse", ref + "^{commit}"], text=True).strip()
        if origin != url or dirty or current != wanted:
            raise RuntimeError("Refusing to overwrite changed or mismatched checkout: " + str(destination))
    if dry_run:
        return ref
    return subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()


# FAISS builds one library per SIMD level; the most specialized one is used.
SIMD_SUFFIXES = {"generic": ("",), "avx2": ("", "_avx2"), "avx512": ("", "_avx2", "_avx512")}


def faiss_library_name(opt_level):
    return "faiss" + SIMD_SUFFIXES[opt_level][-1]


def configure_cpu(source, build, *, python_enabled, jobs, dry_run, opt_level="generic"):
    run(["cmake", "-S", source, "-B", build,
         "-DFAISS_ENABLE_GPU=OFF", "-DFAISS_ENABLE_PYTHON=" + ("ON" if python_enabled else "OFF"),
         "-DFAISS_OPT_LEVEL=" + opt_level, "-DBUILD_TESTING=OFF", "-DBUILD_SHARED_LIBS=ON",
         "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
         "-DPython_EXECUTABLE=" + sys.executable, "-DPython3_EXECUTABLE=" + sys.executable],
        dry_run=dry_run)
    stem = "swigfaiss" if python_enabled else "faiss"
    targets = [part for suffix in SIMD_SUFFIXES[opt_level] for part in ("--target", stem + suffix)]
    run(["cmake", "--build", build, *targets, "--parallel", jobs], dry_run=dry_run)


def smoke_native():
    """Exercise actual build/add/search APIs; import success alone is insufficient."""
    script = """
from cagra_chunksplice.cli import arguments
from cagra_chunksplice.baselines import FaissHNSW, CPUMirage, CuVSCAGRA
import torch
a = arguments(['--mode','external','--external-backend','native','--device','cuda'])
if not torch.cuda.is_available():
    raise RuntimeError('CUDA-enabled PyTorch and a visible GPU are required')
x = torch.randn(256, 32)
q = torch.randn(4, 32)
for index in (FaissHNSW(backend='native'), CPUMirage(backend='native'),
              CuVSCAGRA(a, device='cuda', backend='native')):
    index.build(x).add(x[:32])
    ids = index.search(q, 10)
    index.synchronize()
    assert ids.shape == (4, 10) and index.ntotal == 288
    assert bool(((ids >= -1) & (ids < 288)).all())
    assert index.metadata()['simulated'] is False
    print(index.metadata())
"""
    run([sys.executable, "-c", script])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prefix", type=Path, default=ROOT / ".external")
    parser.add_argument("--cuda-major", type=int, choices=(12, 13))
    parser.add_argument("--cuvs-version", default="25.10.*",
                        help="Pin a release compatible with Python, driver and toolkit.")
    parser.add_argument("--faiss-ref", default="v1.10.0")
    parser.add_argument("--mirage-ref", default="main", help="Use a commit hash for reproducible builds.")
    parser.add_argument("--cuvs-ref", default="branch-25.10")
    parser.add_argument("--source-cuvs", action="store_true",
                        help="Build cuVS/libcuvs Python components from upstream instead of wheels.")
    parser.add_argument("--faiss-opt-level", choices=tuple(SIMD_SUFFIXES), default="generic",
                        help="SIMD level for CPU FAISS and MIRAGE. generic is portable but slower; "
                             "use the highest level the CPU supports for performance comparisons.")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    args = parser.parse_args(argv)
    environment = detect_environment()
    print(json.dumps(environment, indent=2), flush=True)
    if args.probe:
        return 0
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if not args.dry_run:
        if not environment["native_platform_supported"]:
            parser.error(environment["guidance"])
        missing = [name for name in ("git", "cmake", "c++", "swig") if not environment["tools"][name]]
        if missing:
            parser.error("Missing build tools: " + ", ".join(missing))
        if not environment["gpu"]["ok"]:
            parser.error("NVIDIA GPU/driver is not visible through nvidia-smi")
    cuda = args.cuda_major or environment["cuda_toolkit_major"]
    if cuda not in (12, 13):
        parser.error("Detect a CUDA 12/13 toolkit or pass --cuda-major matching the installed toolkit")
    if environment["cuda_toolkit_major"] and cuda != environment["cuda_toolkit_major"]:
        parser.error("--cuda-major does not match nvcc")
    prefix = args.prefix.expanduser().resolve()
    if not args.dry_run:
        (prefix / "src").mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "numpy", "setuptools", "wheel", "packaging"], dry_run=args.dry_run)
    faiss_source, mirage_source = prefix / "src/faiss", prefix / "src/mirage"
    revisions = {
        "faiss": checkout("https://github.com/facebookresearch/faiss.git", args.faiss_ref,
                          faiss_source, args.dry_run),
        "mirage": checkout("https://github.com/dsg-uwaterloo/mirage.git", args.mirage_ref,
                           mirage_source, args.dry_run),
    }
    faiss_build, mirage_build = prefix / "build/faiss", prefix / "build/mirage"
    configure_cpu(faiss_source, faiss_build, python_enabled=True, jobs=args.jobs,
                  dry_run=args.dry_run, opt_level=args.faiss_opt_level)
    run([sys.executable, "-m", "pip", "install", "--no-build-isolation", faiss_build / "faiss/python"], dry_run=args.dry_run)
    configure_cpu(mirage_source, mirage_build, python_enabled=False, jobs=args.jobs,
                  dry_run=args.dry_run, opt_level=args.faiss_opt_level)
    bridge_build = prefix / "build/mirage_bridge"
    run(["cmake", "-S", ROOT / "native/mirage_bridge", "-B", bridge_build,
         "-DMIRAGE_SOURCE=" + str(mirage_source), "-DMIRAGE_BUILD=" + str(mirage_build),
         "-DMIRAGE_FAISS_NAME=" + faiss_library_name(args.faiss_opt_level),
         "-DMIRAGE_REVISION=" + revisions["mirage"], "-DCMAKE_BUILD_TYPE=Release"], dry_run=args.dry_run)
    run(["cmake", "--build", bridge_build, "--parallel", args.jobs], dry_run=args.dry_run)
    if args.source_cuvs:
        if not args.dry_run and not environment["tools"]["nvcc"]:
            parser.error("Source cuVS builds require nvcc")
        source = prefix / "src/cuvs"
        revisions["cuvs"] = checkout("https://github.com/rapidsai/cuvs.git", args.cuvs_ref, source, args.dry_run)
        run(["bash", "build.sh", "libcuvs", "cuvs"], cwd=source, dry_run=args.dry_run)
    else:
        run([sys.executable, "-m", "pip", "install", "--extra-index-url", "https://pypi.nvidia.com",
             f"cuvs-cu{cuda}=={args.cuvs_version}", f"cupy-cuda{cuda}x"], dry_run=args.dry_run)
    run([sys.executable, "-m", "pip", "install", "-e", str(ROOT) + "[test,datasets]"], dry_run=args.dry_run)
    library = bridge_build / "libchunksplice_mirage.so"
    env_file = prefix / "env.sh"
    if args.dry_run:
        print("Export CAGRA_MIRAGE_LIBRARY=" + str(library))
        return 0
    env_file.write_text("export CAGRA_MIRAGE_LIBRARY=" + shlex.quote(str(library)) + "\n", encoding="utf-8")
    manifest = {"environment": environment, "upstream_revisions": revisions,
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "smoke_passed": False, "native_external_comparison_complete": False}
    manifest_path = prefix / "build-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    os.environ["CAGRA_MIRAGE_LIBRARY"] = str(library)
    try:
        smoke_native()
    except (subprocess.CalledProcessError, OSError) as exc:
        manifest["smoke_error"] = str(exc)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    manifest["smoke_passed"] = True
    manifest["installed_packages"] = capture([sys.executable, "-m", "pip", "freeze"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Native API smoke passed. A full Promotion Gate run is still required.")
    print("source " + shlex.quote(str(env_file)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print("Native setup failed: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
