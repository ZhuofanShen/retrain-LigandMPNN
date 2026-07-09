import torch, os, sys, traceback
from torch.utils.data import DataLoader
from model_data_utils import PDBDataset, LengthAwareSubsetBatchSampler, collate_fn


def main():
    nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    torch.multiprocessing.set_start_method('spawn')
    data_path = "dataset/debug"
    files = [f for f in os.listdir(data_path) if f.endswith(".pt")]
    ds = PDBDataset(files, data_path=data_path, max_length=10000)
    sampler = LengthAwareSubsetBatchSampler(ds, token_limit=1200, num_examples_per_epoch=1000000)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=nworkers)
    print(f"--- iterating (num_workers={nworkers}, spawn, CUDA tensors in collate_fn) ---", flush=True)
    try:
        for i, fd in enumerate(loader):
            print(f"batch {i}: S={tuple(fd['S'].shape)} device={fd['S'].device}", flush=True)
        print("DONE OK", flush=True)
    except Exception as e:
        print("EXCEPTION:", type(e).__name__, str(e)[:300], flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
