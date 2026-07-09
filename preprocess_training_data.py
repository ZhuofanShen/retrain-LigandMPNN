import argparse
import json
import os
import signal
import multiprocessing as mp

import torch

from data_utils import parse_PDB


class _TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TimeoutError()


def _init_worker():
    signal.signal(signal.SIGALRM, _alarm_handler)


def process_one(task):
    pdb_id, cif_dir, out_dir, ext, overwrite, timeout = task
    out_plain = os.path.join(out_dir, pdb_id + ".pt")
    out_ligand = os.path.join(out_dir, pdb_id + "_ligand.pt")
    if not overwrite and (os.path.isfile(out_plain) or os.path.isfile(out_ligand)):
        return "skip", pdb_id

    cif_path = os.path.join(cif_dir, pdb_id + ext)
    if not os.path.isfile(cif_path):
        return "missing", pdb_id

    signal.alarm(timeout)
    try:
        pdb_dict = parse_PDB(cif_path, device="cpu", trim_his_tag=True)
    except _TimeoutError:
        return "timeout", pdb_id
    except Exception:
        return "error", pdb_id
    finally:
        signal.alarm(0)
    if pdb_dict is None:
        return "void", pdb_id

    suffix = "_ligand" if pdb_dict["Y"].shape[0] > 0 else ""
    final = os.path.join(out_dir, pdb_id + suffix + ".pt")
    tmp = final + ".tmp"
    torch.save(pdb_dict, tmp)
    os.replace(tmp, final)  # atomic: a concurrent reader never sees a partial .pt
    return "ok", pdb_id


def main():
    parser = argparse.ArgumentParser(
        description="Convert a split of mmCIF/PDB structures into training-readable .pt files."
    )
    parser.add_argument("json", type=str, help="split json file: a list of PDB ids")
    parser.add_argument("--cif_dir", type=str, default=None,
                        help="directory holding <id><ext> structures (default: <json_dir>/mmcif_files)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="output directory for .pt files (default: json path without .json)")
    parser.add_argument("--ext", type=str, default=".cif", help="structure file extension")
    parser.add_argument("--cpus", type=int, default=os.cpu_count(), help="worker processes")
    parser.add_argument("--timeout", type=int, default=120,
                        help="per-structure parse timeout (s); prevents a pathological structure "
                             "from hanging a worker and stalling the whole pool")
    parser.add_argument("--overwrite", action="store_true", help="re-process ids whose .pt already exists")
    args = parser.parse_args()

    pdb_list = json.load(open(args.json, "r"))
    json_dir = os.path.dirname(os.path.abspath(args.json))
    cif_dir = args.cif_dir or os.path.join(json_dir, "mmcif_files")
    out_dir = args.out_dir or args.json[:-5]
    os.makedirs(out_dir, exist_ok=True)

    tasks = [(pdb, cif_dir, out_dir, args.ext, args.overwrite, args.timeout) for pdb in pdb_list]
    n = len(tasks)
    counts = {"ok": 0, "skip": 0, "missing": 0, "void": 0, "error": 0, "timeout": 0}
    omitted = []

    print(f"preprocessing {n} ids from {args.json}\n  cif_dir={cif_dir}\n  out_dir={out_dir}\n  "
          f"cpus={args.cpus} timeout={args.timeout}s", flush=True)
    with mp.Pool(args.cpus, initializer=_init_worker) as pool:
        for i, (status, pdb_id) in enumerate(pool.imap_unordered(process_one, tasks, chunksize=16), 1):
            counts[status] += 1
            if status in ("missing", "void", "error", "timeout"):
                omitted.append(pdb_id)
            if i % 5000 == 0 or i == n:
                print(f"  [{i}/{n}] " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)

    with open(args.json[:-5] + ".omitted_pdb.json", "w") as f:
        json.dump(sorted(omitted), f)
    print(f"done {args.json}: {counts}\n  wrote .pt to {out_dir}\n  omitted ids -> {args.json[:-5]}.omitted_pdb.json",
          flush=True)


if __name__ == "__main__":
    main()
