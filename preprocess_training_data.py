import argparse
import json
import os
import torch

from data_utils import parse_PDB


parser = argparse.ArgumentParser()
parser.add_argument('json', type=str)
args = parser.parse_args()

pdb_list = json.load(open(args.json, "r"))
data_path = args.json[:-5]
with open(data_path + ".omitted_pdb.json", "w") as pf:
    pf.write("[")
with open(data_path + ".void_asmb.json", "w") as pf2:
    pf2.write("[")
with open(data_path + ".redundant_asmb.json", "w") as pf3:
    pf3.write("[")

if not os.path.isdir(data_path):
    os.mkdir(data_path)

for pdb in pdb_list:
    pdb_path = os.path.join(data_path + "_pdb", pdb + ".pdb")
    if not os.path.isfile(pdb_path):
        with open(data_path + ".omitted_pdb.json", "a") as pf:
            pf.write('"' + pdb + '", ')
        continue

    chains_clusters = list()
    with open(pdb_path, "r") as pf:
        biological_assembly = list()
        if_remark_350 = False
        for line in pf:
            if line.startswith("REMARK 350 APPLY THE FOLLOWING TO CHAINS:") or \
                line.startswith("REMARK 350                    AND CHAINS:"):
                if_remark_350 = True
                biological_assembly.extend(line[41:].strip().strip(',').split(', '))
            elif len(biological_assembly) > 0:
                    redundant = False
                    for biological_assembly_2 in chains_clusters:
                        if biological_assembly == biological_assembly_2:
                            redundant = True
                    if not redundant:
                        chains_clusters.append(set(biological_assembly))
                    biological_assembly = list()
            if if_remark_350 and not line.startswith("REMARK 350"):
                break

    nonredundant_chains_clusters = list()
    for i, biological_assembly in enumerate(chains_clusters):
        redundant_chains = set()
        for j, biological_assembly_2 in enumerate(chains_clusters):
            if j != i and biological_assembly_2.issubset(biological_assembly):
                redundant_chains = redundant_chains.union(biological_assembly_2)
        if len(redundant_chains) > 0:
            with open(data_path + ".redundant_asmb.json", "a") as pf3:
                pf3.write('"' + pdb + "_" + "".join(biological_assembly) + '", ')
            biological_assembly = biological_assembly - redundant_chains
        if len(biological_assembly) > 0:
            nonredundant_chains_clusters.append(biological_assembly)

    if len(nonredundant_chains_clusters) == 0:
        try:
            pdb_dict = parse_PDB(pdb_path, device='cpu', trim_his_tag=True)
        except:
            with open(data_path + ".omitted_pdb.json", "a") as pf:
                pf.write('"' + pdb + '", ')
            continue
        suffix = str()
        if pdb_dict is None:
            with open(data_path + ".void_asmb.json", "a") as pf2:
                pf2.write('"' + pdb + suffix + '", ')
            continue
        if pdb_dict["Y"].shape[0] > 0:
            suffix = "_ligand"
        torch.save(pdb_dict, os.path.join(data_path, pdb + ".pt"))
    else:
        for biological_assembly in nonredundant_chains_clusters:
            try:
                pdb_dict = parse_PDB(pdb_path, device='cpu', chains=biological_assembly, trim_his_tag=True)
            except:
                with open(data_path + ".omitted_pdb.json", "a") as pf:
                    pf.write('"' + pdb + '", ')
                continue
            suffix = str()
            if len(nonredundant_chains_clusters) > 1:
                suffix = "_" + "".join(biological_assembly)
            if pdb_dict is None:
                with open(data_path + ".void_asmb.json", "a") as pf2:
                    pf2.write('"' + pdb + suffix + '", ')
                continue
            if pdb_dict["Y"].shape[0] > 0:
                suffix += "_ligand"
            torch.save(pdb_dict, os.path.join(data_path, pdb + suffix + ".pt"))

with open(data_path + ".omitted_pdb.json", "a") as pf:
    pf.write("]")
with open(data_path + ".void_asmb.json", "a") as pf2:
    pf2.write("]")
with open(data_path + ".redundant_asmb.json", "a") as pf3:
    pf3.write("]")
