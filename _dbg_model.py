import torch, os, sys, traceback
import torch.utils.checkpoint  # model.py uses torch.utils.checkpoint.checkpoint without importing it
from torch.utils.data import DataLoader
from model_data_utils import PDBDataset, LengthAwareSubsetBatchSampler, collate_fn, featurize_ligand_neighbors
from model import ProteinMPNN, loss_smoothed, loss_nll


def main():
    model_type = sys.argv[1] if len(sys.argv) > 1 else "ligand_mpnn_new"
    device = torch.device("cuda")
    data_path = "dataset/debug"
    files = [f for f in os.listdir(data_path) if f.endswith(".pt")]
    ds = PDBDataset(files, data_path=data_path, max_length=10000)
    sampler = LengthAwareSubsetBatchSampler(ds, token_limit=1200, num_examples_per_epoch=1000000)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)

    model = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128,
                        num_encoder_layers=3, num_decoder_layers=3, k_neighbors=32,
                        augment_eps=0.2, dropout=0.1, model_type=model_type,
                        atom_context_num=25, device=device)
    model.to(device)
    print(f"=== model_type={model_type} ===", flush=True)
    for i, fd in enumerate(loader):
        try:
            fd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in fd.items()}
            if model_type.startswith("ligand_mpnn"):
                fd = featurize_ligand_neighbors(fd, num_context_atoms=25)
            log_probs = model(fd)
            S = fd["S"]; mask = fd["mask"]
            _, loss = loss_smoothed(S, log_probs, mask)
            print(f"batch {i}: log_probs={tuple(log_probs.shape)} loss={loss.item():.4f}", flush=True)
        except Exception as e:
            print(f"batch {i} EXCEPTION: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
