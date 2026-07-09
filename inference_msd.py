#!/usr/bin/env python
"""Multi-state design (MSD) inference for LigandMPNN.

Designs a single tied sequence that is stabilized on an arbitrary set of "positive"
state structures and destabilized on an arbitrary set of "negative" state structures
(same length, same residue numbering). Per-position objective:

    A_fav(a)   = T_p * logsumexp_{s in positive}( logP_s(a) / T_p )
    A_unfav(a) = T_p * logsumexp_{s in negative}( logP_s(a) / T_p )
    comp(a)    = (1 + lam) * A_fav(a) - lam * A_unfav(a)

lam=0 recovers single-state (positive-ensemble) design; lam=1 is the P_fav^2 / P_unfav
partition objective. logP_s are the autoregressive conditional log-probs from LigandMPNN,
combined at every decoding step with the tied (shared) sequence context.

Supersedes inference_msd_tied.py.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from data_utils import parse_PDB, restype_str_to_int, restype_int_to_str
from model import ProteinMPNN

LIGAND_TYPES = ("ligand_mpnn", "ligand_mpnn_new")
BIG_NEG = 1e8


def residue_ids(out, icodes):
    chain_letters = out["chain_letters"]
    R_idx = out["R_idx"].cpu().numpy()
    return [f"{chain_letters[i]}_{int(R_idx[i])}_{icodes[i]}" for i in range(len(R_idx))]


def cb_from_backbone(X):
    Ca, N, C = X[:, :, 1, :], X[:, :, 0, :], X[:, :, 2, :]
    b, c = Ca - N, C - Ca
    a = torch.cross(b, c, dim=-1)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca


def featurize_states(pdb_paths, device, chains, parse_zero_occ, M, cutoff):
    """Parse every state PDB and stack into one batched feature_dict [B, N, ...].

    Requires identical residue ordering across states (same scaffold, overlaid ligand);
    raises if they disagree so misaligned inputs fail loudly instead of silently.
    """
    states, rid_lists = [], []
    for p in pdb_paths:
        out, _bb, _other, icodes, _wat = parse_PDB(
            p, device=str(device), chains=chains,
            parse_all_atoms=True, parse_atoms_with_zero_occupancy=parse_zero_occ,
            return_prody=True,
        )
        states.append(out)
        rid_lists.append(residue_ids(out, icodes))

    ref = rid_lists[0]
    for p, rids in zip(pdb_paths[1:], rid_lists[1:]):
        if rids != ref:
            raise ValueError(
                f"Residue ordering of {p} differs from {pdb_paths[0]}. MSD requires the "
                f"same length and residue numbering across all states ({len(rids)} vs {len(ref)})."
            )

    B, N = len(states), len(ref)
    X = torch.stack([s["X"] for s in states], 0)
    S = torch.stack([s["S"].long() for s in states], 0)
    mask = torch.stack([s["mask"].float() for s in states], 0)
    R_idx = torch.stack([s["R_idx"] for s in states], 0)
    chain_labels = torch.stack([s["chain_labels"] for s in states], 0)
    xyz_37 = torch.stack([s["xyz_37"] for s in states], 0)
    xyz_37_m = torch.stack([s["xyz_37_m"] for s in states], 0)

    Cb = cb_from_backbone(X)
    Y = torch.zeros((B, N, M, 3), device=device)
    Y_t = torch.zeros((B, N, M), device=device, dtype=torch.int32)
    Y_m = torch.zeros((B, N, M), device=device)
    mask_XY = torch.zeros((B, N), device=device)

    for bi, s in enumerate(states):
        Yg, Ytg, Ymg = s["Y"], s["Y_t"], s["Y_m"].float()
        if Yg.ndim != 2 or Yg.shape[0] == 0:  # apo state
            Yg = torch.zeros((1, 3), device=device)
            Ytg = torch.zeros((1,), device=device, dtype=torch.int32)
            Ymg = torch.zeros((1,), device=device)
        A = Yg.shape[0]
        if A < M:
            Yg = torch.cat([Yg, torch.zeros((M - A, 3), device=device)], 0)
            Ytg = torch.cat([Ytg, torch.zeros((M - A,), device=device, dtype=torch.int32)], 0)
            Ymg = torch.cat([Ymg, torch.zeros((M - A,), device=device)], 0)
        d = torch.sqrt(((Cb[bi, :, None, :] - Yg[None]) ** 2).sum(-1) + 1e-8)  # [N, A]
        nn = torch.topk(d, M, dim=-1, largest=False).indices  # [N, M]
        Y[bi], Y_t[bi], Y_m[bi] = Yg[nn], Ytg[nn], Ymg[nn]
        within = (d <= cutoff).float() * Ymg[None]
        mask_XY[bi] = torch.clamp(within.sum(-1), 0.0, 1.0)

    fd = {
        "X": X, "S": S, "mask": mask, "R_idx": R_idx, "chain_labels": chain_labels,
        "xyz_37": xyz_37, "xyz_37_m": xyz_37_m,
        "Y": Y, "Y_t": Y_t, "Y_m": Y_m, "mask_XY": mask_XY,
        "chain_mask": torch.ones((B, N), device=device),
    }
    return fd, ref


def encode_states(model, fd):
    with torch.no_grad():
        if model.model_type in LIGAND_TYPES:
            E_context, E, E_idx, Y_nodes, Y_edges, Y_m = model.features(fd)
        else:
            E, E_idx = model.features(fd)
        h_V, h_E = model.protein_encode(E, E_idx, fd["mask"])
        if model.model_type == "ligand_mpnn":
            h_V = model.protein_ligand_encode(Y_nodes, Y_edges, Y_m, h_V, E_context, fd["mask"])
    return E_idx, h_V, h_E


def build_decode_mask(E_idx, design_mask, valid_mask, seed):
    device = E_idx.device
    B, N, K = E_idx.shape
    g = torch.Generator(device=device).manual_seed(int(seed))
    order_pool = torch.where((design_mask > 0.5) & (valid_mask > 0.5))[0]
    perm = torch.randperm(len(order_pool), generator=g, device=device)
    order = order_pool[perm].tolist()

    rank = torch.full((N,), -1, device=device, dtype=torch.long)
    for t, pos in enumerate(order):
        rank[pos] = t
    ri, rj = rank[:, None], rank[None, :]
    visible = (rj < 0) | ((ri >= 0) & (rj >= 0) & (rj < ri))  # fixed always visible; designed if earlier
    decode_mask = torch.gather(visible.float()[None].repeat(B, 1, 1), 2, E_idx).unsqueeze(-1)
    return order, decode_mask


def partition_energy(logp, idx, T_p):
    """A(a) = T_p * logsumexp_{s in idx}( logp_s(a) / T_p ), over states idx. logp: [B, 21]."""
    return T_p * torch.logsumexp(logp[idx] / T_p, dim=0)


def msd_logits(lp_i, pos_idx, neg_idx, lam, T_p):
    A_fav = partition_energy(lp_i, pos_idx, T_p)
    if len(neg_idx) == 0 or lam == 0.0:
        return A_fav
    A_unfav = partition_energy(lp_i, neg_idx, T_p)
    return (1.0 + lam) * A_fav - lam * A_unfav


def sample_tied(model, fd, pos_idx, neg_idx, lam, T_p, design_mask, temperature,
                bias, omit, seed, greedy):
    device = fd["X"].device
    B, N = fd["S"].shape
    S = fd["S"][0:1].repeat(B, 1).clone()  # tie initial sequence across states
    E_idx, h_V, h_E = encode_states(model, fd)
    order, decode_mask = build_decode_mask(E_idx, design_mask, fd["mask"][0], seed)

    comp_logits = torch.zeros((N, 21), device=device)
    probs_out = torch.zeros((N, 21), device=device)
    for pos in order:
        with torch.no_grad():
            lp = model.decode(S, h_V, h_E, E_idx, decode_mask, fd["mask"])  # [B, N, 21]
        comp = msd_logits(lp[:, pos, :], pos_idx, neg_idx, lam, T_p)
        comp = comp + bias[pos] - BIG_NEG * omit[pos]
        comp_logits[pos] = comp
        if greedy:
            aa = int(torch.argmax(comp))
        else:
            aa = int(torch.multinomial(F.softmax(comp / temperature, dim=-1), 1))
        S[:, pos] = aa
        probs_out[pos] = F.softmax(comp / temperature, dim=-1)
    return S[0].clone(), comp_logits, probs_out, order


def make_bias_omit(N, device, bias_AA, omit_AA):
    bias = torch.zeros((N, 21), device=device)
    omit = torch.zeros((N, 21), device=device)
    for it in [x.strip() for x in bias_AA.split(",") if x.strip()]:
        aa, val = it.split(":")
        bias[:, restype_str_to_int[aa.strip().upper()]] += float(val)
    for aa in omit_AA.strip().upper():
        if aa in restype_str_to_int:
            omit[:, restype_str_to_int[aa]] = 1.0
    return bias, omit


def seq_to_str(S):
    return "".join(restype_int_to_str[int(x)] for x in S.cpu().numpy())


def split_paths(s):
    out = []
    for chunk in s.split(","):
        out.extend(x for x in chunk.split() if x)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--positive_pdbs", required=True, help="Comma/space separated positive (desired) state PDBs.")
    p.add_argument("--negative_pdbs", default="", help="Comma/space separated negative (undesired) state PDBs.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model_type", default="ligand_mpnn", choices=["ligand_mpnn", "protein_mpnn"])
    p.add_argument("--chains", default="", help="Comma-separated chain IDs to parse (default: all).")
    p.add_argument("--parse_atoms_with_zero_occupancy", action="store_true")
    p.add_argument("--ligand_mpnn_cutoff_for_score", type=float, default=8.0)

    p.add_argument("--selectivity_weight", type=float, default=0.5,
                   help="lam: weight on (A_fav - A_unfav). 0=single-state, 1=P_fav^2/P_unfav.")
    p.add_argument("--partition_temperature", type=float, default=0.2,
                   help="T_p inside the per-ensemble logsumexp (=1/beta); small => best conformer dominates.")
    p.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature on comp.")
    p.add_argument("--num_seqs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--greedy", action="store_true")

    p.add_argument("--fixed_residues", default="", help="Space-separated residue IDs to fix, e.g. 'A_12_ A_70_'.")
    p.add_argument("--bias_AA", default="", help="Comma-separated AA:val, e.g. 'A:0.5,G:-0.2'.")
    p.add_argument("--omit_AA", default="", help="String of globally omitted AAs, e.g. 'CP'.")

    p.add_argument("--out_folder", default="msd_outputs")
    p.add_argument("--name", default="design")
    p.add_argument("--save_pssm", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_pdbs = split_paths(args.positive_pdbs)
    neg_pdbs = split_paths(args.negative_pdbs)
    all_pdbs = pos_pdbs + neg_pdbs
    if not pos_pdbs:
        raise ValueError("--positive_pdbs must contain at least one PDB.")
    chains = [c.strip() for c in args.chains.split(",") if c.strip()] or None

    ck = torch.load(args.checkpoint, map_location=device)
    M = int(ck["atom_context_num"]) if args.model_type in LIGAND_TYPES else 1
    model = ProteinMPNN(
        node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, k_neighbors=int(ck["num_edges"]),
        device=device, atom_context_num=M, model_type=args.model_type,
        ligand_mpnn_use_side_chain_context=False,
    )
    model.load_state_dict(ck["model_state_dict"]); model.to(device).eval()

    fd, rids = featurize_states(all_pdbs, device, chains, args.parse_atoms_with_zero_occupancy,
                                M, args.ligand_mpnn_cutoff_for_score)
    B, N = fd["S"].shape
    pos_idx = list(range(len(pos_pdbs)))
    neg_idx = list(range(len(pos_pdbs), B))

    design_mask = torch.ones((N,), device=device)
    fixed = set(args.fixed_residues.split())
    for i, rid in enumerate(rids):
        if rid in fixed:
            design_mask[i] = 0.0
    bias, omit = make_bias_omit(N, device, args.bias_AA, args.omit_AA)

    os.makedirs(args.out_folder, exist_ok=True)
    records, dbg = [], []
    for t in range(args.num_seqs):
        seed = args.seed + t
        S_single, comp0, probs0, order = sample_tied(
            model, fd, pos_idx, neg_idx, 0.0, args.partition_temperature,
            design_mask, args.temperature, bias, omit, seed, args.greedy)
        S_msd, compL, probsL, _ = sample_tied(
            model, fd, pos_idx, neg_idx, args.selectivity_weight, args.partition_temperature,
            design_mask, args.temperature, bias, omit, seed, args.greedy)
        single_str, msd_str = seq_to_str(S_single), seq_to_str(S_msd)
        n_diff = sum(a != b for a, b in zip(single_str, msd_str))
        records.append((t, single_str, msd_str, n_diff))
        dbg.append({"comp_single": comp0.cpu(), "comp_msd": compL.cpu(),
                    "probs_single": probs0.cpu(), "probs_msd": probsL.cpu(),
                    "decoding_order": torch.tensor(order)})
        print(f"[seq {t}] single<->msd differ at {n_diff}/{int(design_mask.sum())} designed positions")

    fasta = os.path.join(args.out_folder, f"{args.name}.fa")
    with open(fasta, "w") as f:
        meta = f"lam={args.selectivity_weight} T_p={args.partition_temperature} T={args.temperature} pos={len(pos_pdbs)} neg={len(neg_pdbs)}"
        for t, single_str, msd_str, n_diff in records:
            f.write(f">{args.name}_{t}_single {meta}\n{single_str}\n")
            f.write(f">{args.name}_{t}_msd {meta} n_diff={n_diff}\n{msd_str}\n")

    torch.save({
        "all_pdbs": all_pdbs, "positive_pdbs": pos_pdbs, "negative_pdbs": neg_pdbs,
        "residue_ids": rids, "design_mask": design_mask.cpu(), "mask_XY": fd["mask_XY"].cpu(),
        "selectivity_weight": args.selectivity_weight, "partition_temperature": args.partition_temperature,
        "records": [(t, s, m, d) for t, s, m, d in records], "debug": dbg,
    }, os.path.join(args.out_folder, f"{args.name}.pt"))

    if args.save_pssm:
        try:
            save_pssm(probsL, rids, design_mask, os.path.join(args.out_folder, f"{args.name}"))
        except Exception as e:
            print(f"PSSM plot skipped ({e}); install matplotlib to enable it.")
    print(f"Wrote {fasta}")


def save_pssm(probs, rids, design_mask, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = design_mask.cpu().numpy() > 0.5
    mat = probs.cpu().numpy()[sel].T  # [21, Nsel]
    labels = [rid for rid, k in zip(rids, sel) if k]
    plt.figure(figsize=(max(10, 0.18 * mat.shape[1]), 6))
    plt.imshow(mat, aspect="auto"); plt.colorbar(label="P(AA)")
    plt.yticks(range(21), [restype_int_to_str[i] for i in range(21)])
    step = max(1, mat.shape[1] // 40)
    plt.xticks(range(0, mat.shape[1], step), labels[::step], rotation=90, fontsize=6)
    plt.title("MSD per-position amino-acid probability"); plt.tight_layout()
    plt.savefig(prefix + ".pssm.png", dpi=200); plt.close()


if __name__ == "__main__":
    main()
