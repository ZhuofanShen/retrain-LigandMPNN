import os, sys, warnings, numpy as np, torch
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from model_data_utils import PDBDataset, LengthAwareSubsetBatchSampler, collate_fn, featurize_ligand_neighbors
from model import ProteinMPNN, loss_smoothed


def check_sampler_ordering():
    print("\n[1] sampler keeps subset length-sorted (synthetic 2000 proteins)")
    class FakeDS:
        def __init__(self, lengths): self.lengths = lengths
        def __len__(self): return len(self.lengths)
    np.random.seed(1)
    lengths = list(np.random.randint(40, 1000, size=2000))
    ds = FakeDS(lengths)
    s = LengthAwareSubsetBatchSampler(ds, token_limit=8000, num_examples_per_epoch=1000000)
    sel = np.array([lengths[i] for i in s.indices])
    sorted_ok = bool(np.all(np.diff(sel) >= 0))
    real = pad = 0
    for b in s.batches:
        Ls = np.array([lengths[i] for i in b]); real += Ls.sum(); pad += Ls.max() * len(b)
    print(f"    indices sorted by length: {sorted_ok}")
    print(f"    batches={len(s.batches)} padding_waste={100*(pad-real)/pad:.1f}%")


def check_collate_cpu_and_workers():
    print("\n[2] collate returns CPU tensors; fork workers, no CUDA-IPC warning")
    dp = "dataset/debug"
    files = [f for f in os.listdir(dp) if f.endswith(".pt")]
    ds = PDBDataset(files, data_path=dp, max_length=10000)
    s = LengthAwareSubsetBatchSampler(ds, token_limit=1200, num_examples_per_epoch=1000000)
    loader = DataLoader(ds, batch_sampler=s, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fd = next(iter(loader))
    devs = {str(v.device) for v in fd.values() if torch.is_tensor(v)}
    pinned = all(v.is_pinned() for v in fd.values() if torch.is_tensor(v))
    print(f"    collate output devices: {devs}  pinned={pinned}")
    ipc = [str(w.message) for w in caught if "CUDA" in str(w.message)]
    print(f"    CUDA-IPC warnings: {len(ipc)}")


def check_grads_flow():
    print("\n[3] grads actually flow (ligand_mpnn, mixed precision)")
    device = torch.device("cuda")
    dp = "dataset/debug"
    files = [f for f in os.listdir(dp) if f.endswith(".pt")]
    ds = PDBDataset(files, data_path=dp, max_length=10000)
    s = LengthAwareSubsetBatchSampler(ds, token_limit=1200, num_examples_per_epoch=1000000)
    loader = DataLoader(ds, batch_sampler=s, collate_fn=collate_fn, num_workers=0)
    model = ProteinMPNN(model_type="ligand_mpnn", atom_context_num=25, augment_eps=0.2,
                        dropout=0.1, device=device).to(device)
    fd = next(iter(loader))
    fd = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in fd.items()}
    fd = featurize_ligand_neighbors(fd, num_context_atoms=25)  # build [B,L,M,*] graph on-device
    with torch.cuda.amp.autocast():
        lp = model(fd)
        _, loss = loss_smoothed(fd["S"], lp, fd["mask"])
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in model.parameters())
    print(f"    loss={loss.item():.4f}  params_with_nonzero_grad={n_with_grad}/{n_total}")


if __name__ == "__main__":
    check_sampler_ordering()
    check_collate_cpu_and_workers()
    check_grads_flow()
    print("\nVERIFY DONE")
