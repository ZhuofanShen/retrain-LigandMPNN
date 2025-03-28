import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )

class PDBDataset(Dataset):
    def __init__(self, PDB_pts, data_path):
        self.PDB_pts = PDB_pts
        self.data_path = data_path

    def __len__(self):
        return len(self.PDB_pts)

    def __getitem__(self, index):
        return torch.load(os.path.join(self.data_path, self.PDB_pts[index]))


def worker_init_fn(worker_id):
    np.random.seed()

def get_pdbs(data_loader, max_length=10000, num_units=1000000):
    pdb_dict_list = list()
    for pdb_dict in data_loader:
        if len(pdb_dict['S']) <= max_length:
            pdb_dict["X"] = torch.squeeze(pdb_dict["X"], 0)
            pdb_dict["mask"] = torch.squeeze(pdb_dict["mask"], 0)
            pdb_dict["Y"] = torch.squeeze(pdb_dict["Y"], 0)
            pdb_dict["Y_t"] = torch.squeeze(pdb_dict["Y_t"], 0)
            pdb_dict["Y_m"] = torch.squeeze(pdb_dict["Y_m"], 0)
            pdb_dict["R_idx"] = torch.squeeze(pdb_dict["R_idx"], 0)
            pdb_dict["chain_labels"] = torch.squeeze(pdb_dict["chain_labels"], 0)
            pdb_dict["S"] = torch.squeeze(pdb_dict["S"], 0)
            pdb_dict["xyz_37"] = torch.squeeze(pdb_dict["xyz_37"], 0)
            pdb_dict["xyz_37_m"] = torch.squeeze(pdb_dict["xyz_37_m"], 0)
            pdb_dict_list.append(pdb_dict)
        if len(pdb_dict_list) >= num_units:
            break
    return pdb_dict_list

class Batches:
    def __init__(self, pdb_dict_list, max_length=10000, batch_size=1000000):
        self.dataset = pdb_dict_list
        self.lengths = [len(pdb_dict['S']) for pdb_dict in self.dataset]
        self.batch_size = batch_size

        # Cluster into batches of similar sizes
        batch_list = list()
        batch = list()
        batch_length = 1
        for idx in np.argsort(self.lengths):
            length = self.lengths[idx]
            if length <= max_length:
                if length * batch_length <= self.batch_size:
                    batch.append(idx)
                    batch_length += 1
                else:
                    batch_list.append(batch)
                    batch = list()
                    batch_length = 1
        if batch_length > 1:
            batch_list.append(batch)
        self.batch_list = batch_list

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self):
        np.random.shuffle(self.batch_list)
        for b_idx in self.batch_list:
            batch = [self.dataset[i] for i in b_idx]
            yield batch

def batch_featurize(
    batch,
    device,
    cutoff_for_score=8.0,
    use_atom_context=True,
    n_nbh_ligand_atoms=25,
    model_type="ligand_mpnn"
):
    # alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    B = len(batch)
    L_max = max([b["S"].shape[0] for b in batch])
    # l_ligand_max = max([b["Y"].shape[0] for b in batch]) # + 1
    l_ligand_squeezed_max = 0

    xyz_37_batch = torch.zeros([B, L_max, 37, 3], dtype=torch.float32, device=device) #[B,L,37,3] - xyz coordinates for all atoms if needed
    xyz_37_m_batch = torch.zeros([B, L_max, 37], dtype=torch.int32, device=device) #[B,L,37] - mask for all coords
    # Y_batch = torch.zeros([B, l_ligand_max, 3], dtype=torch.float32, device=device) #[B,l,3] - for ligandMPNN coords
    # Y_t_batch = torch.zeros([B, l_ligand_max], dtype=torch.int32, device=device) #[B,l] - element type
    # Y_m_batch = torch.zeros([B, l_ligand_max], dtype=torch.int32, device=device) #[B,l] - mask
    Y_nbh_batch = torch.zeros([B, L_max, n_nbh_ligand_atoms, 3], dtype=torch.float32, device=device) #[B,L,M,3] - for ligandMPNN coords
    Y_t_nbh_batch = torch.zeros([B, L_max, n_nbh_ligand_atoms], dtype=torch.int32, device=device) #[B,L,M] - element type
    Y_m_nbh_batch = torch.zeros([B, L_max, n_nbh_ligand_atoms], dtype=torch.int32, device=device) #[B,L,M] - mask
    X_batch = torch.zeros([B, L_max, 4, 3], dtype=torch.float32, device=device) #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O
    S_batch = torch.zeros([B, L_max], dtype=torch.long, device=device) #[B,L] - integer protein sequence encoded using "restype_STRtoINT"
    R_idx_batch = torch.zeros([B, L_max], dtype=torch.long, device=device) #[B,L] - primary sequence residue index
    mask_batch = torch.zeros([B, L_max], dtype=torch.int32, device=device) #[B,L] - mask for missing regions - should be removed! all ones most of the time
    nn_idx_batch = torch.zeros([B, L_max, n_nbh_ligand_atoms], dtype=torch.long, device=device) #[B,L,M]
    # Y_scale_batch = torch.ones([B, l_ligand_max], dtype=torch.int32, device=device) #[B,l] - Y_scale at least is 1
    Y_scale_squeezed_list = list()
    # mask_XY_batch = torch.zeros([B, L_max], dtype=torch.int32, device=device) #[B,L]
    chain_labels_batch = torch.zeros([B, L_max], dtype=torch.long, device=device) #[B,L] - integer labels for chain letters
    chain_mask_batch = torch.zeros([B, L_max], dtype=torch.long, device=device) #[B,L]

    feature_dict = dict()
    feature_dict["batch_size"] = B
    for i_pdb, input_dict in enumerate(batch):
        # if model_type == "ligand_mpnn":
        l = input_dict["S"].shape[0]
        l_ligand = input_dict["Y"].shape[0]
        pad_n_residues = L_max - l
        # pad_n_ligand_atoms = l_ligand_max - l_ligand
        pad_n_nbh_ligand_atoms = n_nbh_ligand_atoms - l_ligand
        if pad_n_nbh_ligand_atoms < 0:
            pad_n_nbh_ligand_atoms = 0
        mask = input_dict["mask"].to(device) # [L]
        Y = input_dict["Y"].to(device) # [l, 3]
        Y_t = input_dict["Y_t"].to(device) # [l]
        Y_m = input_dict["Y_m"].to(device) # [l]
        X = input_dict["X"].to(device)
        N = X[:, 0, :] # [L, 3]
        CA = X[:, 1, :] # [L, 3]
        C = X[:, 2, :] # [L, 3]
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA # [L, 3]
        mask_CBY = mask[:, None] * Y_m[None, :] # [L, M]
        try:
            Y[None, :, :]
        except:
            Y = torch.zeros([1, 3], dtype=torch.float32, device=device)
            Y_t = torch.zeros([1], dtype=torch.int32, device=device)
            Y_m = torch.zeros([1], dtype=torch.int32, device=device)
        L2_AB = torch.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1) # [L, l]
        L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0 # [L, l]
        nn_idx = torch.argsort(L2_AB, -1)[:, :n_nbh_ligand_atoms] # [L, M], nn means nearest neighbor
        Y_nbh = gather_context_atom_features(Y, nn_idx)
        Y_t_nbh = gather_context_atoms(Y_t, nn_idx)
        Y_m_nbh = gather_context_atoms(Y_m, nn_idx)
        # L2_AB_nn = torch.gather(L2_AB, 1, nn_idx) # [L, M]
        # D_XY = torch.sqrt(L2_AB_nn[:, 0]) # D_AB_closest [L]
        # mask_XY = (D_XY < cutoff_for_score) * mask * Y_m_nbh[:, 0] # [L], residues have context atoms or not
        # mask_XY_batch[i_pdb,:] = F.pad(mask_XY, (0, pad_n_residues), mode="constant", value=0)
        Y_mask_scattered = torch.zeros([l, l_ligand], dtype=torch.int32, device=device) # [L, l]
        Y_scale_unsqueezed = torch.sum(Y_mask_scattered.scatter_(
            1, nn_idx, torch.ones_like(nn_idx, dtype=torch.int32, device=device)
            ), dim=0) # [l]
        # squeeze ligand atoms without neighborhood residues
        nn_idx_squeezed_scattered = [-1] * l_ligand
        Y_scale_squeezed = list()
        for i_atom, scale in enumerate(Y_scale_unsqueezed):
            if scale > 0:
                nn_idx_squeezed_scattered[i_atom] = len(Y_scale_squeezed)
                Y_scale_squeezed.append(scale)
        nn_idx_squeezed_scattered = torch.tensor(nn_idx_squeezed_scattered, dtype=torch.int32, device=device)
        nn_idx_squeezed = gather_context_atoms(nn_idx_squeezed_scattered, nn_idx)
        nn_idx_batch[i_pdb,:,:] = F.pad(nn_idx_squeezed, (0, pad_n_nbh_ligand_atoms, 0, pad_n_residues), mode="constant", value=0)
        Y_scale_squeezed = torch.tensor(Y_scale_squeezed, dtype=torch.int32, device=device)
        # Y_scale_batch[i_pdb,:] = F.pad(Y_scale, (0, pad_n_ligand_atoms), mode="constant", value=1) # Y_scale at least is 1
        if l_ligand_squeezed_max < Y_scale_squeezed.shape[0]:
            l_ligand_squeezed_max = Y_scale_squeezed.shape[0]
        Y_scale_squeezed_list.append(Y_scale_squeezed)
        # if "side_chain_mask" in list(input_dict):
        #     output_dict["side_chain_mask"] = input_dict["side_chain_mask"][None,].to(device)
        # Y_batch[i_pdb,:,:] = F.pad(Y, (0, 0, 0, pad_n_ligand_atoms), mode="constant", value=0)
        # Y_t_batch[i_pdb,:] = F.pad(Y_t, (0, pad_n_ligand_atoms), mode="constant", value=0)
        # if use_atom_context:
        #     Y_m_batch[i_pdb,:] = F.pad(Y_m, (0, pad_n_ligand_atoms), mode="constant", value=0)
        Y_nbh_batch[i_pdb,:,:,:] = F.pad(Y_nbh, (0, 0, 0, pad_n_nbh_ligand_atoms, 0, pad_n_residues), mode="constant", value=0)
        Y_t_nbh_batch[i_pdb,:,:] = F.pad(Y_t_nbh, (0, pad_n_nbh_ligand_atoms, 0, pad_n_residues), mode="constant", value=0)
        if use_atom_context:
            Y_m_nbh_batch[i_pdb,:,:] = F.pad(Y_m_nbh, (0, pad_n_nbh_ligand_atoms, 0, pad_n_residues), mode="constant", value=0)

        R_idx_list = list()
        count = 0
        R_idx_prev = -100000
        for R_index in list(input_dict["R_idx"]):
            if R_idx_prev == R_index:
                count += 1
            R_idx_list.append(R_index + count)
            R_idx_prev = R_index
        R_idx_renumbered = torch.tensor(R_idx_list, device=device)
        R_idx_batch[i_pdb,:] = F.pad(R_idx_renumbered, (0, pad_n_residues), mode="constant", value=0)
        # output_dict["R_idx_original"] = input_dict["R_idx"][None,].to(device)
        chain_labels = input_dict["chain_labels"].to(device)
        chain_labels_batch[i_pdb,:] = F.pad(chain_labels, (0, pad_n_residues), mode="constant", value=0)
        S = input_dict["S"].to(device)
        S_batch[i_pdb,:] = F.pad(S, (0, pad_n_residues), mode="constant", value=0)
        # chain_mask = input_dict["chain_mask"].to(device)
        chain_mask = torch.ones((l))
        chain_mask_batch[i_pdb,:] = F.pad(chain_mask, (0, pad_n_residues), mode="constant", value=0)
        mask = input_dict["mask"].to(device)
        mask_batch[i_pdb,:] = F.pad(mask, (0, pad_n_residues), mode="constant", value=0)
        X = input_dict["X"].to(device)
        X_batch[i_pdb,:,:,:] = F.pad(X, (0, 0, 0, 0, 0, pad_n_residues), mode="constant", value=0)
        xyz_37 = input_dict.get("xyz_37")
        if xyz_37 is not None:
            xyz_37 = xyz_37.to(device)
            xyz_37_batch[i_pdb,:,:,:] = F.pad(xyz_37, (0, 0, 0, 0, 0, pad_n_residues), mode="constant", value=0)
            xyz_37_m = input_dict["xyz_37_m"].to(device)
            xyz_37_m_batch[i_pdb,:,:] = F.pad(xyz_37_m, (0, 0, 0, pad_n_residues), mode="constant", value=0)
    Y_scale_batch = torch.ones([B, l_ligand_squeezed_max], dtype=torch.int32, device=device) #[B,l] - Y_scale at least is 1
    for Y_scale_squeezed in Y_scale_squeezed_list:
        pad_n_ligand_atoms = l_ligand_squeezed_max - Y_scale_squeezed.shape[0]
        Y_scale_batch[i_pdb,:] = F.pad(Y_scale_squeezed, (0, pad_n_ligand_atoms), mode="constant", value=1) # Y_scale at least is 1
    feature_dict["xyz_37"] = xyz_37_batch #[B,L,37,3] - xyz coordinates for all atoms if needed
    feature_dict["xyz_37_m"] = xyz_37_m_batch #[B,L,37] - mask for all coords
    feature_dict["X"] = X_batch #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O
    feature_dict["S"] = S_batch # [B,L] - integer protein sequence encoded using "restype_STRtoINT"
    feature_dict["R_idx"] = R_idx_batch #[B,L] - primary sequence residue index
    feature_dict["mask"] = mask_batch # [B,L] - mask for missing regions - should be removed! all ones most of the time
    feature_dict["chain_labels"] = chain_labels_batch #[B,L] - integer labels for chain letters
    feature_dict["chain_mask"] = chain_mask_batch #[B,L]
    # feature_dict["Y"] = Y_batch #[B,l,3] - for ligandMPNN coords
    # feature_dict["Y_t"] = Y_t_batch #[B,l] - element type
    # feature_dict["Y_m"] = Y_m_batch #[B,l] - mask
    feature_dict["Y"] = Y_nbh_batch #[B,L,M,3] - for ligandMPNN coords
    feature_dict["Y_t"] = Y_t_nbh_batch #[B,L,M] - element type
    feature_dict["Y_m"] = Y_m_nbh_batch #[B,L,M] - mask
    feature_dict["nn_idx"] = nn_idx_batch #[B,L,M]
    feature_dict["Y_scale"] = Y_scale_batch #[B,l]
    # feature_dict["mask_XY"] = mask_XY_batch #[B,L]
    return feature_dict

def gather_context_atom_features(Y, nn_idx):
    # Y [l, C] at Neighbor indices [L, M] => Y [L, M, C]
    Y_r = Y[None, :, :].repeat(nn_idx.shape[0], 1, 1) # [L, l, C]
    Y_tmp = torch.gather(Y_r, 1, nn_idx[:, :, None].repeat(1, 1, Y.shape[-1]))
    Y = torch.zeros(
        [nn_idx.shape[0], nn_idx.shape[1], Y.shape[-1]], dtype=torch.float32, device=Y.device
    )
    Y[:, :Y_tmp.shape[1]] = Y_tmp
    return Y

def gather_context_atoms(Y_t, nn_idx):
    # Y_t [l] at Neighbor indices [L, M] => Y [L, M]
    Y_t_r = Y_t[None, :].repeat(nn_idx.shape[0], 1) # [L, l]
    Y_t_tmp = torch.gather(Y_t_r, 1, nn_idx)
    Y_t = torch.zeros(
        [nn_idx.shape[0], nn_idx.shape[1]], dtype=torch.int32, device=Y_t.device
    )
    Y_t[:, :Y_t_tmp.shape[1]] = Y_t_tmp
    return Y_t


side_chain_atom_types = [
    6,
    6,
    6,
    8,
    8,
    16,
    6,
    6,
    6,
    7,
    7,
    8,
    8,
    16,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    8,
    8,
    6,
    7,
    7,
    8,
    6,
    6,
    6,
    7,
    8,
]

periodic_table_features = [
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
    ],
    [
        0,
        1,
        18,
        1,
        2,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        1,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    ],
    [
        0,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
    ],
]
