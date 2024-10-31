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
omitted_list = list()
parsed_list = list()

for pdb in pdb_list:
    pdb_path = os.path.join(data_path, pdb + ".pdb")
    if os.path.isfile(pdb_path):
        try:
            pdb_dict = parse_PDB(pdb_path, device='cuda')
            torch.save(pdb_dict, os.path.join(data_path, pdb + ".pt"))
        except:
            pass
    else:
        omitted_list.append(pdb)

if len(omitted_list) > 0:
    json.dump(omitted_list, open(data_path + ".omitted.json", "w"))
