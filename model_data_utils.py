import json
import numpy as np
import time
import torch
from torch.utils.data import Dataset


def batch_featurize(
    batch,
    cutoff_for_score=8.0,
    use_atom_context=True,
    number_of_ligand_atoms=30,
    # model_type="ligand_mpnn",
):
    # alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    B = len(batch)
    L_max = max([b["S"].shape[0] for b in batch])
    # l_ligand_max = max([b["Y"].shape[0] for b in batch]) # + 1
    xyz_37_batch = np.zeros([B, L_max, 37, 3]) #[B,L,37,3] - xyz coordinates for all atoms if needed
    xyz_37_m_batch = np.zeros([B, L_max, 37], dtype=np.int32) #[B,L,37] - mask for all coords
    # Y_batch = np.zeros([B, l_ligand_max, 3]) #[B,l_max,3] - for ligandMPNN coords
    # Y_t_batch = np.zeros([B, l_ligand_max], dtype=np.int32) #[B,l_max] - element type
    # Y_m_batch = np.zeros([B, l_ligand_max], dtype=np.int32) #[B,l_max] - mask
    Y_nbh_batch = np.zeros([B, L_max, number_of_ligand_atoms, 3]) #[B,L,30,3] - for ligandMPNN coords
    Y_t_nbh_batch = np.zeros([B, L_max, number_of_ligand_atoms], dtype=np.int32) #[B,L,30] - element type
    Y_m_nbh_batch = np.zeros([B, L_max, number_of_ligand_atoms], dtype=np.int32) #[B,L,30] - mask
    X_batch = np.zeros([B, L_max, 4, 3]) #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O
    S_batch = np.zeros([B, L_max], dtype=np.int32) #[B,L] - integer protein sequence encoded using "restype_STRtoINT"
    R_idx_batch = np.zeros([B, L_max], dtype=np.int32) #[B,L] - primary sequence residue index
    mask_batch = np.zeros([B, L_max], dtype=np.int32) #[B,L] - mask for missing regions - should be removed! all ones most of the time
    nn_idx_batch = np.zeros([B, L_max, 30], dtype=np.int32) #[B,L,30]
    mask_XY_batch = np.zeros([B, L_max], dtype=np.int32) #[B,L]
    chain_labels_batch = np.zeros([B, L_max], dtype=np.int32) #[B,L] - integer labels for chain letters
    chain_mask_batch = np.zeros([B, L_max], dtype=np.int32) #[B,L]

    feature_dict = dict()
    feature_dict["batch_size"] = B
    for i, input_dict in enumerate(batch):
        # if model_type == "ligand_mpnn":
        l = input_dict["S"].shape[0]
        l_ligand = input_dict["Y"].shape[0]
        pad_n_residues = L_max - l
        # pad_n_ligand_atoms = l_ligand_max - l_ligand
        # if pad_n_ligand_atoms < 0:
        #     pad_n_ligand_atoms = 0
        pad_n_nbh_ligand_atoms = number_of_ligand_atoms - l_ligand
        if pad_n_nbh_ligand_atoms < 0:
            pad_n_nbh_ligand_atoms = 0
        mask = input_dict["mask"] # [L]
        Y = input_dict["Y"] # [l, 3]
        Y_t = input_dict["Y_t"] # [l]
        Y_m = input_dict["Y_m"] # [l]
        N = input_dict["X"][:, 0, :]
        CA = input_dict["X"][:, 1, :]
        C = input_dict["X"][:, 2, :]
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA # [L, 3]
        mask_CBY = mask[:, None] * Y_m[None, :] # [L, 30]
        L2_AB = torch.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1) # [L, 30]
        L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0 # [L, 30]
        nn_idx = torch.argsort(L2_AB, -1)[:, :number_of_ligand_atoms] # [L, 30] nn means nearest neighbor
        nn_idx_batch[i,:,:] = np.pad(nn_idx, [[0,pad_n_residues], [0,0]], "constant", constant_values=(0, ))
        L2_AB_nn = torch.gather(L2_AB, 1, nn_idx) # [L, 30]
        D_XY = torch.sqrt(L2_AB_nn[:, 0]) # D_AB_closest [L]
        Y_nbh = gather_context_atom_features(Y, nn_idx)
        Y_t_nbh = gather_context_atoms(Y_t, nn_idx)
        Y_m_nbh = gather_context_atoms(Y_m, nn_idx)
        mask_XY = (D_XY < cutoff_for_score) * mask * Y_m_nbh[:, 0] # [L], whether a residue has context atoms or not
        mask_XY_batch[i,:] = np.pad(mask_XY, [[0,pad_n_residues]], "constant", constant_values=(0, ))
        # if "side_chain_mask" in list(input_dict):
        #     output_dict["side_chain_mask"] = input_dict["side_chain_mask"][None,]
        # Y_batch[i,:,:] = np.pad(Y, [[0,pad_n_ligand_atoms], [0,0]], "constant", constant_values=(0, ))
        # Y_t_batch[i,:] = np.pad(Y_t, [[0,pad_n_ligand_atoms]], "constant", constant_values=(0, ))
        # if use_atom_context:
        #     Y_m_batch[i,:] = np.pad(Y_m, [[0,pad_n_ligand_atoms]], "constant", constant_values=(0, ))
        Y_nbh_batch[i,:,:,:] = np.pad(Y_nbh, [[0,pad_n_residues], [0,pad_n_nbh_ligand_atoms], [0,0]], "constant", constant_values=(0, ))
        Y_t_nbh_batch[i,:,:] = np.pad(Y_t_nbh, [[0,pad_n_residues], [0,pad_n_nbh_ligand_atoms]], "constant", constant_values=(0, ))
        if use_atom_context:
            Y_m_nbh_batch[i,:,:] = np.pad(Y_m_nbh, [[0,pad_n_residues], [0,pad_n_nbh_ligand_atoms]], "constant", constant_values=(0, ))

        R_idx_list = []
        count = 0
        R_idx_prev = -100000
        for R_idx in list(input_dict["R_idx"]):
            if R_idx_prev == R_idx:
                count += 1
            R_idx_list.append(R_idx + count)
            R_idx_prev = R_idx
        # R_idx_renumbered = torch.tensor(R_idx_list, device=R_idx.device)
        R_idx_renumbered = np.array(R_idx_list)
        R_idx_batch[i,:] = np.pad(R_idx_renumbered, [[0,pad_n_residues]], "constant", constant_values=(0, ))
        # output_dict["R_idx_original"] = input_dict["R_idx"][None,]
        chain_labels = input_dict["chain_labels"]
        chain_labels_batch[i,:] = np.pad(chain_labels, [[0,pad_n_residues]], "constant", constant_values=(0, ))
        S = input_dict["S"]
        S_batch[i,:] = np.pad(S, [[0,pad_n_residues]], "constant", constant_values=(0, ))
        # chain_mask = input_dict["chain_mask"]
        chain_mask = np.ones((l))
        chain_mask_batch[i,:] = np.pad(chain_mask, [[0,pad_n_residues]], "constant", constant_values=(0, ))
        mask = input_dict["mask"]
        mask_batch[i,:] = np.pad(mask, [[0,pad_n_residues]], "constant", constant_values=(0, ))
        X = input_dict["X"]
        X_pad = np.pad(X, [[0,pad_n_residues], [0,0], [0,0]], "constant", constant_values=(0, ))
        X_batch[i,:,:,:] = X_pad
        xyz_37 = input_dict.get("xyz_37")
        if xyz_37 is not None:
            xyz_37_pad = np.pad(xyz_37, [[0,pad_n_residues], [0,0], [0,0]], "constant", constant_values=(0, ))
            xyz_37_batch[i,:,:,:] = xyz_37_pad
            xyz_37_m = input_dict.get("xyz_37_m")
            xyz_37_m_pad = np.pad(xyz_37_m, [[0,pad_n_residues], [0,0]], "constant", constant_values=(0, ))
            xyz_37_m_batch[i,:,:] = xyz_37_m_pad
    feature_dict["xyz_37"] = torch.from_numpy(xyz_37_batch).to(dtype=torch.float32, device=R_idx.device) #[B,L,37,3] - xyz coordinates for all atoms if needed
    feature_dict["xyz_37_m"] = torch.from_numpy(xyz_37_m_batch).to(dtype=torch.float32, device=R_idx.device) #[B,L,37] - mask for all coords
    # feature_dict["Y"] = torch.from_numpy(Y_batch).to(dtype=torch.float32, device=R_idx.device) #[B,l_max,3] - for ligandMPNN coords
    # feature_dict["Y_t"] = torch.from_numpy(Y_t_batch).to(dtype=torch.int32, device=R_idx.device) #[B,l_max] - element type
    # feature_dict["Y_m"] = torch.from_numpy(Y_m_batch).to(dtype=torch.float32, device=R_idx.device) #[B,l_max] - mask
    feature_dict["Y"] = torch.from_numpy(Y_nbh_batch).to(dtype=torch.float32, device=R_idx.device) #[B,L,30,3] - for ligandMPNN coords
    feature_dict["Y_t"] = torch.from_numpy(Y_t_nbh_batch).to(dtype=torch.int32, device=R_idx.device) #[B,L,30] - element type
    feature_dict["Y_m"] = torch.from_numpy(Y_m_nbh_batch).to(dtype=torch.float32, device=R_idx.device) #[B,L,30] - mask
    feature_dict["X"] = torch.from_numpy(X_batch).to(dtype=torch.float32, device=R_idx.device) #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O
    feature_dict["S"] = torch.from_numpy(S_batch).to(dtype=torch.long, device=R_idx.device) # [B,L] - integer protein sequence encoded using "restype_STRtoINT"
    feature_dict["R_idx"] = torch.from_numpy(R_idx_batch).to(dtype=torch.long, device=R_idx.device) #[B,L] - primary sequence residue index
    feature_dict["mask"] = torch.from_numpy(mask_batch).to(dtype=torch.float32, device=R_idx.device) # [B,L] - mask for missing regions - should be removed! all ones most of the time
    feature_dict["nn_idx"] = torch.from_numpy(nn_idx_batch).to(dtype=torch.long, device=R_idx.device) #[B,L,30]
    feature_dict["mask_XY"] = torch.from_numpy(mask_XY_batch).to(dtype=torch.float32, device=R_idx.device) #[B,L]
    feature_dict["chain_labels"] = torch.from_numpy(chain_labels_batch).to(dtype=torch.long, device=R_idx.device) #[B,L] - integer labels for chain letters
    feature_dict["chain_mask"] = torch.from_numpy(chain_mask_batch).to(dtype=torch.long, device=R_idx.device) #[B,L]
    return feature_dict

def gather_context_atom_features(Y, nn_idx):
    # Y [l_max, C] at Neighbor indices [L, M] => Y [L, M, C]
    Y_r = Y[None, :, :].repeat(nn_idx.shape[0], 1, 1) # [L, l_max, C]
    Y_tmp = torch.gather(Y_r, 1, nn_idx[:, :, None].repeat(1, 1, Y.shape[-1]))
    Y = torch.zeros(
        [nn_idx.shape[0], nn_idx.shape[1], Y.shape[-1]], dtype=torch.float32, device=Y.device
    )
    Y[:, :Y_tmp.shape[1]] = Y_tmp
    return Y

def gather_context_atoms(Y_t, nn_idx):
    # Y_t [l_max] at Neighbor indices [L, M] => Y [L, M]
    Y_t_r = Y_t[None, :].repeat(nn_idx.shape[0], 1) # [L, l_max]
    Y_t_tmp = torch.gather(Y_t_r, 1, nn_idx)
    Y_t = torch.zeros(
        [nn_idx.shape[0], nn_idx.shape[1]], dtype=torch.int32, device=Y_t.device
    )
    Y_t[:, :Y_t_tmp.shape[1]] = Y_t_tmp
    return Y_t


class PDB_Dataset(Dataset):
    def __init__(self, PDB_IDs, data_path, loader):
        self.PDB_IDs = PDB_IDs
        self.data_path = data_path
        self.loader = loader

    def __len__(self):
        return len(self.PDB_IDs)

    def __getitem__(self, index):
        return self.loader(self.PDB_IDs[index], self.data_path)

def loader_pdb(PDB_ID, data_path):
    return torch.load(f"{data_path}/{PDB_ID}.pt")

def worker_init_fn(worker_id):
    np.random.seed()

def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    pdb_dict_list = list()
    for _ in range(repeat):
        for pdb_dict in data_loader:
            if len(pdb_dict['S']) <= max_length:
                pdb_dict_list.append(pdb_dict)
            if len(pdb_dict_list) >= num_units:
                break
    return pdb_dict_list


class BatchList:
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
