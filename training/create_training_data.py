from __future__ import print_function

import numpy as np
import torch
import torch.utils
from prody import *

confProDy(verbosity="none")

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}
restype_str_to_int = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}
restype_int_to_str = {
    0: "A",
    1: "C",
    2: "D",
    3: "E",
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",
}
alphabet = list(restype_str_to_int)

element_list = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mb",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Uut",
    "Fl",
    "Uup",
    "Lv",
    "Uus",
    "Uuo",
]
element_list = [item.upper() for item in element_list]
# element_dict = dict(zip(element_list, range(1,len(element_list))))
element_dict_rev = dict(zip(range(1, len(element_list)), element_list))


def get_aligned_coordinates(protein_atoms, CA_dict: dict, atom_name: str):
    """
    protein_atoms: prody atom group
    CA_dict: mapping between chain_residue_idx_icodes and integers
    atom_name: atom to be parsed; e.g. CA
    """
    atom_atoms = protein_atoms.select(f"name {atom_name}")

    if atom_atoms != None:
        atom_coords = atom_atoms.getCoords()
        atom_resnums = atom_atoms.getResnums()
        atom_chain_ids = atom_atoms.getChids()
        atom_icodes = atom_atoms.getIcodes()

    atom_coords_ = np.zeros([len(CA_dict), 3], np.float32)
    atom_coords_m = np.zeros([len(CA_dict)], np.int32)
    if atom_atoms != None:
        for i in range(len(atom_resnums)):
            code = atom_chain_ids[i] + "_" + str(atom_resnums[i]) + "_" + atom_icodes[i]
            if code in list(CA_dict):
                atom_coords_[CA_dict[code], :] = atom_coords[i]
                atom_coords_m[CA_dict[code]] = 1
    return atom_coords_, atom_coords_m

def parse_PDB(
    input_path: str,
    device: str = "cpu",
    chains: list = [],
    parse_all_atoms: bool = False,
    parse_atoms_with_zero_occupancy: bool = False
):
    """
    input_path : path for the input PDB
    device: device for the torch.Tensor
    chains: a list specifying which chains need to be parsed; e.g. ["A", "B"]
    parse_all_atoms: if False parse only N,CA,C,O otherwise all 37 atoms
    parse_atoms_with_zero_occupancy: if True atoms with zero occupancy will be parsed
    """
    element_list = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mb",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Uut",
        "Fl",
        "Uup",
        "Lv",
        "Uus",
        "Uuo",
    ]
    element_list = [item.upper() for item in element_list]
    element_dict = dict(zip(element_list, range(1, len(element_list))))
    restype_3to1 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    restype_STRtoINT = {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
        "X": 20,
    }

    atom_order = {
        "N": 0,
        "CA": 1,
        "C": 2,
        "CB": 3,
        "O": 4,
        "CG": 5,
        "CG1": 6,
        "CG2": 7,
        "OG": 8,
        "OG1": 9,
        "SG": 10,
        "CD": 11,
        "CD1": 12,
        "CD2": 13,
        "ND1": 14,
        "ND2": 15,
        "OD1": 16,
        "OD2": 17,
        "SD": 18,
        "CE": 19,
        "CE1": 20,
        "CE2": 21,
        "CE3": 22,
        "NE": 23,
        "NE1": 24,
        "NE2": 25,
        "OE1": 26,
        "OE2": 27,
        "CH2": 28,
        "NH1": 29,
        "NH2": 30,
        "OH": 31,
        "CZ": 32,
        "CZ2": 33,
        "CZ3": 34,
        "NZ": 35,
        "OXT": 36,
    }

    if not parse_all_atoms:
        atom_types = ["N", "CA", "C", "O"]
    else:
        atom_types = [
            "N",
            "CA",
            "C",
            "CB",
            "O",
            "CG",
            "CG1",
            "CG2",
            "OG",
            "OG1",
            "SG",
            "CD",
            "CD1",
            "CD2",
            "ND1",
            "ND2",
            "OD1",
            "OD2",
            "SD",
            "CE",
            "CE1",
            "CE2",
            "CE3",
            "NE",
            "NE1",
            "NE2",
            "OE1",
            "OE2",
            "CH2",
            "NH1",
            "NH2",
            "OH",
            "CZ",
            "CZ2",
            "CZ3",
            "NZ",
        ]

    atoms = parsePDB(input_path)
    if not parse_atoms_with_zero_occupancy:
        atoms = atoms.select("occupancy > 0")
    if chains:
        str_out = ""
        for item in chains:
            str_out += " chain " + item + " or"
        atoms = atoms.select(str_out[1:-3])

    protein_atoms = atoms.select("protein")
    backbone = protein_atoms.select("backbone")
    other_atoms = atoms.select("not protein and not water")
    water_atoms = atoms.select("water")

    CA_atoms = protein_atoms.select("name CA")
    CA_resnums = CA_atoms.getResnums()
    CA_chain_ids = CA_atoms.getChids()
    CA_icodes = CA_atoms.getIcodes()

    CA_dict = {}
    for i in range(len(CA_resnums)):
        code = CA_chain_ids[i] + "_" + str(CA_resnums[i]) + "_" + CA_icodes[i]
        CA_dict[code] = i

    xyz_37 = np.zeros([len(CA_dict), 37, 3], np.float32)
    xyz_37_m = np.zeros([len(CA_dict), 37], np.int32)
    for atom_name in atom_types:
        xyz, xyz_m = get_aligned_coordinates(protein_atoms, CA_dict, atom_name)
        xyz_37[:, atom_order[atom_name], :] = xyz
        xyz_37_m[:, atom_order[atom_name]] = xyz_m

    N = xyz_37[:, atom_order["N"], :]
    CA = xyz_37[:, atom_order["CA"], :]
    C = xyz_37[:, atom_order["C"], :]
    O = xyz_37[:, atom_order["O"], :]

    N_m = xyz_37_m[:, atom_order["N"]]
    CA_m = xyz_37_m[:, atom_order["CA"]]
    C_m = xyz_37_m[:, atom_order["C"]]
    O_m = xyz_37_m[:, atom_order["O"]]

    mask = N_m * CA_m * C_m * O_m  # must all 4 atoms exist

    b = CA - N
    c = C - CA
    a = np.cross(b, c, axis=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

    chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
    R_idx = np.array(CA_resnums, dtype=np.int32)
    S = CA_atoms.getResnames()
    S = [restype_3to1[AA] if AA in list(restype_3to1) else "X" for AA in list(S)]
    S = np.array([restype_STRtoINT[AA] for AA in list(S)], np.int32)
    X = np.concatenate([N[:, None], CA[:, None], C[:, None], O[:, None]], 1)

    try:
        Y = np.array(other_atoms.getCoords(), dtype=np.float32)
        Y_t = list(other_atoms.getElements())
        Y_t = np.array(
            [
                element_dict[y_t.upper()] if y_t.upper() in element_list else 0
                for y_t in Y_t
            ],
            dtype=np.int32,
        )
        Y_m = (Y_t != 1) * (Y_t != 0)

        Y = Y[Y_m, :]
        Y_t = Y_t[Y_m]
        Y_m = Y_m[Y_m]
    except:
        Y = np.zeros([1, 3], np.float32)
        Y_t = np.zeros([1], np.int32)
        Y_m = np.zeros([1], np.int32)

    output_dict = {}
    output_dict["X"] = torch.tensor(X, device=device, dtype=torch.float32)
    output_dict["mask"] = torch.tensor(mask, device=device, dtype=torch.int32)
    output_dict["Y"] = torch.tensor(Y, device=device, dtype=torch.float32)
    output_dict["Y_t"] = torch.tensor(Y_t, device=device, dtype=torch.int32)
    output_dict["Y_m"] = torch.tensor(Y_m, device=device, dtype=torch.int32)

    output_dict["R_idx"] = torch.tensor(R_idx, device=device, dtype=torch.int32)
    output_dict["chain_labels"] = torch.tensor(
        chain_labels, device=device, dtype=torch.int32
    )

    output_dict["chain_letters"] = CA_chain_ids

    mask_c = []
    chain_list = list(set(output_dict["chain_letters"]))
    chain_list.sort()
    for chain in chain_list:
        mask_c.append(
            torch.tensor(
                [chain == item for item in output_dict["chain_letters"]],
                device=device,
                dtype=bool,
            )
        )

    output_dict["mask_c"] = mask_c
    output_dict["chain_list"] = chain_list

    output_dict["S"] = torch.tensor(S, device=device, dtype=torch.int32)

    output_dict["xyz_37"] = torch.tensor(xyz_37, device=device, dtype=torch.float32)
    output_dict["xyz_37_m"] = torch.tensor(xyz_37_m, device=device, dtype=torch.int32)

    return output_dict, backbone, other_atoms, CA_icodes, water_atoms


import json, time, os, sys, glob
import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureDataset():
    def __init__(self, jsonl_file, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        with open(jsonl_file) as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq'] 
                name = entry['name']

                # Convert raw coords to np arrays
                #for key, val in entry['coords'].items():
                #    entry['coords'][key] = np.asarray(val)

                # Check if in alphabet
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        if True:
                            self.data.append(entry)
                        else:
                            discard_count['bad_seq_length'] += 1
                    else:
                        discard_count['too_long'] += 1
                else:
                    print(name, bad_chars, entry['seq'])
                    discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            print('discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class StructureDatasetPDB():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            # print(entry)
            # print(entry.keys())
            seq = entry['S']
            # name = entry['name']

            # bad_chars = set([s for s in seq]).difference(alphabet_set)
            # if len(bad_chars) == 0:
            if seq.shape[0] <= max_length:
                self.data.append(entry)
            else:
                discard_count['too_long'] += 1
            # else:
            #     #print(name, bad_chars, entry['seq'])
            #     discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                #print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            #print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


    
class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
        collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch

def get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms):
    device = CB.device
    mask_CBY = mask[:, None] * Y_m[None, :] # [L, n_ligand_atoms]
    L2_AB = torch.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1) # [L, n_ligand_atoms]
    L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0 # [L, n_ligand_atoms]

    nn_idx = torch.argsort(L2_AB, -1)[:, :number_of_ligand_atoms] # [L, n_ligand_atoms] nn means nearest neighbor
    L2_AB_nn = torch.gather(L2_AB, 1, nn_idx) # [L, n_ligand_atoms]
    D_AB_closest = torch.sqrt(L2_AB_nn[:, 0]) # [L]

    Y_r = Y[None, :, :].repeat(CB.shape[0], 1, 1)
    Y_t_r = Y_t[None, :].repeat(CB.shape[0], 1)
    Y_m_r = Y_m[None, :].repeat(CB.shape[0], 1)

    Y_tmp = torch.gather(Y_r, 1, nn_idx[:, :, None].repeat(1, 1, 3))
    Y_t_tmp = torch.gather(Y_t_r, 1, nn_idx)
    Y_m_tmp = torch.gather(Y_m_r, 1, nn_idx)

    Y = torch.zeros(
        [CB.shape[0], number_of_ligand_atoms, 3], dtype=torch.float32, device=device
    )
    Y_t = torch.zeros(
        [CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device
    )
    Y_m = torch.zeros(
        [CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device
    )

    num_nn_update = Y_tmp.shape[1]
    Y[:, :num_nn_update] = Y_tmp
    Y_t[:, :num_nn_update] = Y_t_tmp
    Y_m[:, :num_nn_update] = Y_m_tmp

    return Y, Y_t, Y_m, D_AB_closest

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
    # L_ligand_max = max([b["Y"].shape[0] for b in batch]) # + 1

    xyz_37_batch = np.zeros([B, L_max, 37, 3]) #[B,L,37,3] - xyz coordinates for all atoms if needed
    xyz_37_m_batch = np.zeros([B, L_max, 37], dtype=np.int32) #[B,L,37] - mask for all coords
    Y_batch = np.zeros([B, L_max, number_of_ligand_atoms, 3]) #[B,L,num_context_atoms,3] - for ligandMPNN coords
    Y_t_batch = np.zeros([B, L_max, number_of_ligand_atoms], dtype=np.int32) #[B,L,num_context_atoms] - element type
    Y_m_batch = np.zeros([B, L_max, number_of_ligand_atoms], dtype=np.int32) #[B,L,num_context_atoms] - mask
    X_batch = np.zeros([B, L_max, 4, 3]) #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O
    S_batch = np.zeros([B, L_max], dtype=np.int32) #[B,L] - integer protein sequence encoded using "restype_STRtoINT"
    R_idx_batch = np.zeros([B, L_max], dtype=np.int32) #[B,L] - primary sequence residue index
    mask_batch = np.zeros([B, L_max], dtype=np.int32) #[B,L] - mask for missing regions - should be removed! all ones most of the time
    mask_XY_batch = np.zeros([B, L_max], dtype=np.int32) #[B,L]
    chain_labels_batch = np.zeros([B, L_max], dtype=np.int32) #[B,L] - integer labels for chain letters
    chain_mask_batch = np.zeros([B, L_max], dtype=np.int32) #[B,L]

    feature_dict = dict()
    feature_dict["batch_size"] = B
    for i, input_dict in enumerate(batch):
        # if model_type == "ligand_mpnn":
        l = input_dict["S"].shape[0]
        l_ligand = input_dict["Y"].shape[0]
        mask = input_dict["mask"]
        Y = input_dict["Y"]
        Y_t = input_dict["Y_t"]
        Y_m = input_dict["Y_m"]
        N = input_dict["X"][:, 0, :]
        CA = input_dict["X"][:, 1, :]
        C = input_dict["X"][:, 2, :]
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        Y, Y_t, Y_m, D_XY = get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms)
        mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, 0]
        mask_XY_batch[i,:] = np.pad(mask_XY, [[0,L_max-l]], "constant", constant_values=(0, ))
        # if "side_chain_mask" in list(input_dict):
        #     output_dict["side_chain_mask"] = input_dict["side_chain_mask"][None,]
        pad_n_ligand_atom = number_of_ligand_atoms - l_ligand
        if pad_n_ligand_atom < 0:
            pad_n_ligand_atom = 0
        Y_batch[i,:,:,:] = np.pad(Y, [[0,L_max-l], [0,pad_n_ligand_atom], [0,0]], "constant", constant_values=(0, ))
        Y_t_batch[i,:,:] = np.pad(Y_t, [[0,L_max-l], [0,pad_n_ligand_atom]], "constant", constant_values=(0, ))
        if use_atom_context:
            Y_m_batch[i,:,:] = np.pad(Y_m, [[0,L_max-l], [0,pad_n_ligand_atom]], "constant", constant_values=(0, ))

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
        R_idx_batch[i,:] = np.pad(R_idx_renumbered, [[0,L_max-l]], "constant", constant_values=(0, ))
        # output_dict["R_idx_original"] = input_dict["R_idx"][None,]
        chain_labels = input_dict["chain_labels"]
        chain_labels_batch[i,:] = np.pad(chain_labels, [[0,L_max-l]], "constant", constant_values=(0, ))
        S = input_dict["S"]
        S_batch[i,:] = np.pad(S, [[0,L_max-l]], "constant", constant_values=(0, ))
        # chain_mask = input_dict["chain_mask"]
        chain_mask = np.ones((l))
        chain_mask_batch[i,:] = np.pad(chain_mask, [[0,L_max-l]], "constant", constant_values=(0, ))
        mask = input_dict["mask"]
        mask_batch[i,:] = np.pad(mask, [[0,L_max-l]], "constant", constant_values=(0, ))
        X = input_dict["X"]
        X_pad = np.pad(X, [[0,L_max-l], [0,0], [0,0]], "constant", constant_values=(0, ))
        X_batch[i,:,:,:] = X_pad
        xyz_37 = input_dict.get("xyz_37")
        if xyz_37 is not None:
            xyz_37_pad = np.pad(xyz_37, [[0,L_max-l], [0,0], [0,0]], "constant", constant_values=(0, ))
            xyz_37_batch[i,:,:,:] = xyz_37_pad
            xyz_37_m = input_dict.get("xyz_37_m")
            xyz_37_m_pad = np.pad(xyz_37_m, [[0,L_max-l], [0,0]], "constant", constant_values=(0, ))
            xyz_37_m_batch[i,:,:] = xyz_37_m_pad
    feature_dict["xyz_37"] = torch.from_numpy(xyz_37_batch).to(dtype=torch.float32, device=R_idx.device) #[B,L,37,3] - xyz coordinates for all atoms if needed
    feature_dict["xyz_37_m"] = torch.from_numpy(xyz_37_m_batch).to(dtype=torch.float32, device=R_idx.device) #[B,L,37] - mask for all coords
    feature_dict["Y"] = torch.from_numpy(Y_batch).to(dtype=torch.float32, device=R_idx.device) #[B,L,num_context_atoms,3] - for ligandMPNN coords
    feature_dict["Y_t"] = torch.from_numpy(Y_t_batch).to(dtype=torch.int32, device=R_idx.device) #[B,L,num_context_atoms] - element type
    feature_dict["Y_m"] = torch.from_numpy(Y_m_batch).to(dtype=torch.float32, device=R_idx.device) #[B,L,num_context_atoms] - mask
    feature_dict["X"] = torch.from_numpy(X_batch).to(dtype=torch.float32, device=R_idx.device) #[B,L,4,3] - backbone xyz coordinates for N,CA,C,O
    feature_dict["S"] = torch.from_numpy(S_batch).to(dtype=torch.long, device=R_idx.device) # [B,L] - integer protein sequence encoded using "restype_STRtoINT"
    feature_dict["R_idx"] = torch.from_numpy(R_idx_batch).to(dtype=torch.long, device=R_idx.device) #[B,L] - primary sequence residue index
    feature_dict["mask"] = torch.from_numpy(mask_batch).to(dtype=torch.float32, device=R_idx.device) # [B,L] - mask for missing regions - should be removed! all ones most of the time
    feature_dict["mask_XY"] = torch.from_numpy(mask_XY_batch).to(dtype=torch.float32, device=R_idx.device) #[B,L]
    feature_dict["chain_labels"] = torch.from_numpy(chain_labels_batch).to(dtype=torch.long, device=R_idx.device) #[B,L] - integer labels for chain letters
    feature_dict["chain_mask"] = torch.from_numpy(chain_mask_batch).to(dtype=torch.long, device=R_idx.device) #[B,L]
    return feature_dict


import copy

batch_size = 1
BATCH_COPIES = batch_size

def main(args):
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=30,
        model_type=args.model_type,
    )
    if False:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # need preprocess to .pt files
    protein_dict, backbone, other_atoms, icodes, _ = parse_PDB("7s69_A.pdb", device = "cpu", chains = [], parse_all_atoms = False,
        parse_atoms_with_zero_occupancy = False)
    # torch.save(protein_dict, 'pdb_dict.pt')
    # xyz_37 [N, 37, 3]
    # X [N, 4, 3]
    # xyz_37_m [N, 37]
    # protein_dict['xyz_37_m'][1]
    # tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    #     0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0], dtype=torch.int32)
    # mask [N]
    # R_idx [N]
    # S [N] # sequence AA types
    # chain_labels chain_letters [N]
    # chain_list [n_chains]
    # mask_c [n_chains, N] bool
    # Y [num_ligand_atoms, 3]
    # Y_t [num_ligand_atoms]
    # Y_m [num_ligand_atoms]

    pdb_dict_list = [protein_dict, protein_dict, protein_dict]
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=1000)
    for ix, protein in enumerate(dataset_valid):
        seq_score = {}
#        score_list = []
#        seq_s = []
#        all_probs_list = []
        batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
        # Z, Z_m, Z_t, X, X_m, Y, Y_m, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, tied_beta = \
        #     tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict_in, omit_AA_dict, tied_positions_dict, pssm_dict)
        batch_feature_dict = batch_featurize(batch_clones)
        # torch.save(batch_feature_dict, 'batch_feature_dict.pt')
        output_dict = model.sample(batch_feature_dict)
        loss, loss_per_residue = get_score(
                    output_dict["S"],
                    output_dict["log_probs"],
                    feature_dict["mask"] * feature_dict["chain_mask"],
                )
        loss_XY, _ = get_score(
                    output_dict["S"], output_dict["log_probs"], combined_mask
                )
#         if not args.use_sc:
#             X_m = X_m * 0
#         if not args.use_DNA_RNA:
#             Y_m = Y_m * 0
#         if not args.use_ligand:
#             Z_m = Z_m * 0
#         if args.mask_hydrogen:
#             mask_hydrogen = ~(Z_t == 40)  #1 for not hydrogen, 0 for hydrogen
#             Z_m = Z_m*mask_hydrogen
            
#         #pssm_log_odds_mask = (pssm_log_odds_all > args.pssm_threshold).float() #1.0 for true, 0.0 for false
#         pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false
#         name_ = batch_clones[0]['name']

#         for temp in temperature_use:
#             for j in range(NUM_BATCHES):
#                 randn_2 = torch.randn(chain_M.shape, device=X.device)
#                 if tied_positions_dict == None:
#                     sample_dict = model.sample(X, X_m, Y, Y_m, Z, Z_m, Z_t, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag))
# #                    S_sample = sample_dict["S"] 
#                 else:
#                     sample_dict = model.tied_sample(X, X_m, Y, Y_m, Z, Z_m, Z_t, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), tied_pos=tied_pos_list_of_lists_list[0])
#                 #                            # Compute scores
#                 S_sample = sample_dict["S"]
#                 log_probs = model(X, X_m, Y, Y_m, Z, Z_m, Z_t, randn_2, S_sample, chain_M*chain_M_pos, chain_encoding_all, residue_idx, mask)
#                 mask_for_loss = mask*chain_M*chain_M_pos
#                 scores = _scores(S_sample, log_probs, mask_for_loss)
#                 scores = scores.cpu().data.numpy()
# #                all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
#                 for b_ix in range(BATCH_COPIES):
# #                    masked_chain_length_list = masked_chain_length_list_list[b_ix]
# #                    masked_list = masked_list_list[b_ix]
#                     #seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
#                     seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
#                     score = scores[b_ix]
#                     seq_score[seq] = score
#         if args.save_score:
#             score_file = base_folder + '/scores/' + batch_clones[0]['name'] + '.npy'
#             np.save(score_file, np.array(score_list, np.float32))

if __name__ == "__main__":
    main(args)
