import numpy as np
import torch
from prody import *


confProDy(verbosity="none")

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

atom_types_4 = ["N", "CA", "C", "O"]

atom_types_all = [
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
    chains: list = None,
    parse_all_atoms: bool = False,
    parse_atoms_with_zero_occupancy: bool = False,
    trim_his_tag: bool = False
):
    """
    input_path : path for the input PDB
    device: device for the torch.Tensor
    chains: a list specifying which chains need to be parsed; e.g. ["A", "B"]
    parse_all_atoms: if False parse only N,CA,C,O otherwise all 37 atoms
    parse_atoms_with_zero_occupancy: if True atoms with zero occupancy will be parsed
    """

    if not parse_all_atoms:
        atom_types = atom_types_4
    else:
        atom_types = atom_types_all
    atoms = parsePDB(input_path)
    if not parse_atoms_with_zero_occupancy:
        atoms = atoms.select("occupancy > 0")
    if chains and len(chains) > 0:
        str_out = ""
        for item in chains:
            str_out += " chain " + item + " or"
        atoms = atoms.select(str_out[1:-3])

    protein_atoms = atoms.select("protein")
    # backbone = protein_atoms.select("backbone")
    # other_atoms = atoms.select("not protein and not water")
    # water_atoms = atoms.select("water")

    if protein_atoms is None:
        return None

    CA_atoms = protein_atoms.select("name CA")
    CA_resnums = CA_atoms.getResnums()
    CA_chain_ids = CA_atoms.getChids()
    CA_icodes = CA_atoms.getIcodes()

    CA_dict = dict()
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

    # b = CA - N
    # c = C - CA
    # a = np.cross(b, c, axis=-1)
    # CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

    chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
    R_idx = np.array(CA_resnums, dtype=np.int32)
    seq = CA_atoms.getResnames()
    seq = [restype_3to1[AA] if AA in list(restype_3to1) else "X" for AA in list(seq)]
    S = np.array(["ACDEFGHIKLMNPQRSTVWYX".find(AA) for AA in seq], np.int32) # restype_STRtoINT[AA]
    X = np.concatenate([N[:, None], CA[:, None], C[:, None], O[:, None]], 1)

    if trim_his_tag:
        his_tag_ranges = [0]
        seq = "".join(seq)
        chain_label = -1
        for i, chain_label_i in enumerate(chain_labels):
            if chain_label_i != chain_label:
                chain_label = chain_label_i
                if i > 0:
                    if seq[i-10:i-4] == "HHHHHH":
                        his_tag_ranges.extend([i-10, i])
                    elif seq[i-9:i-3] == "HHHHHH":
                        his_tag_ranges.extend([i-9, i])
                    elif seq[i-8:i-2] == "HHHHHH":
                        his_tag_ranges.extend([i-8, i])
                    elif seq[i-7:i-1] == "HHHHHH":
                        his_tag_ranges.extend([i-7, i])
                    elif seq[i-6:i] == "HHHHHH":
                        his_tag_ranges.extend([i-6, i])
                if seq[i+4:i+10] == "HHHHHH":
                    his_tag_ranges.extend([i, i+10])
                elif seq[i+3:i+9] == "HHHHHH":
                    his_tag_ranges.extend([i, i+9])
                elif seq[i+2:i+8] == "HHHHHH":
                    his_tag_ranges.extend([i, i+8])
                elif seq[i+1:i+7] == "HHHHHH":
                    his_tag_ranges.extend([i, i+7])
                elif seq[i:i+6] == "HHHHHH":
                    his_tag_ranges.extend([i, i+6])
        if seq[i-9:i-3] == "HHHHHH":
            his_tag_ranges.extend([i-9, i+1])
        elif seq[i-8:i-2] == "HHHHHH":
            his_tag_ranges.extend([i-8, i+1])
        elif seq[i-7:i-1] == "HHHHHH":
            his_tag_ranges.extend([i-7, i+1])
        elif seq[i-6:i] == "HHHHHH":
            his_tag_ranges.extend([i-6, i+1])
        elif seq[i-5:i+1] == "HHHHHH":
            his_tag_ranges.extend([i-5, i+1])
        his_tag_ranges.append(i+1)
        X_list = list()
        mask_list = list()
        R_idx_list = list()
        chain_labels_list = list()
        S_list = list()
        xyz_37_list = list()
        xyz_37_m_list = list()
        for i in range(0, len(his_tag_ranges), 2):
            start_index = his_tag_ranges[i]
            end_index = his_tag_ranges[i+1]
            if start_index != end_index:
                X_list.append(X[start_index:end_index, :, :])
                mask_list.append(mask[start_index:end_index])
                R_idx_list.append(R_idx[start_index:end_index])
                chain_labels_list.append(chain_labels[start_index:end_index])
                S_list.append(S[start_index:end_index])
                xyz_37_list.append(xyz_37[start_index:end_index, :, :])
                xyz_37_m_list.append(xyz_37_m[start_index:end_index, :])
        X = np.concatenate(X_list, 0)
        mask = np.concatenate(mask_list, 0)
        R_idx = np.concatenate(R_idx_list, 0)
        chain_labels = np.concatenate(chain_labels_list, 0)
        S = np.concatenate(S_list, 0)
        xyz_37 = np.concatenate(xyz_37_list, 0)
        xyz_37_m = np.concatenate(xyz_37_m_list, 0)

    Y = list()
    Y_t = list()
    for line in open(input_path, "r"):
        if ((line.startswith("ATOM  ") or line.startswith("HETATM")) and \
                not line[17:20] in ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", \
                                    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", \
                                    "UNK", "MSE", "PTR", "CYX", "HID", "HIE", "HIP", \
                                    "EOH", "EDO", "GOL", "PEG", "PG4", "PG5", "1PE", "PG6", \
                                    "FMT", "ACT", " CL", "SO4", "PO4", "WAT", "HOH"] \
                or line.startswith("HETATM") and line[17:20] == "UNK") \
                and len(line) >= 78:
            if chains and line[20:22].strip() not in chains:
                continue
            atom_type = line.rstrip()[-4:].strip().upper()
            if atom_type.endswith("+") or atom_type.endswith("-"):
                atom_type = atom_type[:-2]
            atom_type = element_dict[atom_type]
            if atom_type == 1:
                continue
            Y.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            Y_t.append(atom_type)
    assert len(Y) == len(Y_t)
    Y = np.array(Y, dtype=np.float32)
    Y_t = np.array(Y_t, dtype=np.int32)
    Y_m = (Y_t != 1) * (Y_t != 0)

    output_dict = dict()
    output_dict["X"] = torch.tensor(X, device=device, dtype=torch.float32)
    output_dict["mask"] = torch.tensor(mask, device=device, dtype=torch.int32)
    output_dict["R_idx"] = torch.tensor(R_idx, device=device, dtype=torch.int32)
    output_dict["chain_labels"] = torch.tensor(chain_labels, device=device, dtype=torch.int32)

    # output_dict["chain_letters"] = CA_chain_ids

    # mask_c = list()
    # chain_list = list(set(output_dict["chain_letters"]))
    # chain_list.sort()
    # for chain in chain_list:
    #     mask_c.append(
    #         torch.tensor(
    #             [chain == item for item in output_dict["chain_letters"]],
    #             device=device,
    #             dtype=bool,
    #         )
    #     )

    # output_dict["mask_c"] = mask_c
    # output_dict["chain_list"] = chain_list

    output_dict["S"] = torch.tensor(S, device=device, dtype=torch.int32)
    output_dict["xyz_37"] = torch.tensor(xyz_37, device=device, dtype=torch.float32)
    output_dict["xyz_37_m"] = torch.tensor(xyz_37_m, device=device, dtype=torch.int32)

    output_dict["Y"] = torch.tensor(Y, device=device, dtype=torch.float32)
    output_dict["Y_t"] = torch.tensor(Y_t, device=device, dtype=torch.int32)
    output_dict["Y_m"] = torch.tensor(Y_m, device=device, dtype=torch.int32)

    return output_dict
