import os
from rdkit import Chem

import numpy as np
import pandas as pd
import torch
import dgl
import warnings
from tqdm import tqdm

from dgl.data.chem.utils import smiles_to_bigraph
from dgl.data.chem import CanonicalAtomFeaturizer
from dgl.data.chem import CanonicalBondFeaturizer
from model.BayeshERG_model import BayeshERG
from model.BayeshERG_model import RegularizationAccumulator


from torch.utils.data import DataLoader




from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdmolfiles, rdmolops

import argparse

TRAIN_LEN = 14322 # Number of training data


def warn(*args, **kwargs):
    pass



def collate(graphs):
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(labels)

def load_data(df, atom_featurizer, bond_featurizer):
    print("---------------- Target loading --------------------")
    test_g = [smiles_to_bigraph(smi, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for smi in df['smiles']]
    test_data = list(test_g)
    print("---------------- Target loading complete --------------------")
    return test_data

def load_model(model_path):
    wr = 1e-4 ** 2. / TRAIN_LEN
    dr = 2. / TRAIN_LEN
    reg_acc = RegularizationAccumulator()
    model = BayeshERG(reg_acc=reg_acc,
                          node_input_dim=74,
                          edge_input_dim=12,
                          node_hidden_dim=int(2 ** 7),
                          edge_hidden_dim=int(2 ** 7),
                          num_step_message_passing=7,
                          num_step_mha=1, wr=wr, dr=dr)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    return model

def prediction(model, df, test_data, device, samples = 100):

    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate, drop_last=False)

    with torch.no_grad():
        model.eval()
        score_df = pd.DataFrame(columns=list(range(samples)))
        attention_result = []
        for t in tqdm(range(samples), desc="Sampling"):

            pred_score = []
            num_atom_list = []
            pred_att = []

            for _, bg in enumerate(test_loader):
                
                lengths = bg.batch_num_nodes
                atom_feats = bg.ndata.pop('h').to(device)
                bond_feats = bg.edata.pop('e').to(device)

                pred, w_list = model(bg, atom_feats, bond_feats)
                w_tensor = torch.cat(w_list, dim=1)
                w_tensor = w_tensor.detach().to('cpu').numpy()


                pred_sof = pred[1].detach().to('cpu').numpy()
                pred_sof = np.array(pred_sof).reshape(-1, 2)
                pred_score.append(pred_sof)

                num_atom_list += [x + 1 for x in lengths]
                pred_att.append(w_tensor)


            pred_score = np.vstack(pred_score)
            pred_att = np.hstack(pred_att)
            attention_result.append(pred_att)
            score_df[t] = pd.Series(pred_score[:, 1])

        attention_result = np.stack(attention_result, axis=2)

        class_df = score_df.mean(axis=1)
        mean_temp = pd.concat([class_df] * (samples), axis=1, ignore_index=True)

        alea_df = (score_df * (1 - score_df)).mean(axis=1)
        epis_df = ((score_df - mean_temp) ** 2).mean(axis=1)
        mean_att = np.mean(attention_result, axis=2)

        df['score'] = class_df
        df['alea'] = alea_df
        df['epis'] = epis_df

        return df, num_atom_list, mean_att

def attention_visulaizer(name, df, mean_att, num_atom_list):
    os.mkdir("attention_results/"+name)
    c = 0
    for j, n_atoms in enumerate(num_atom_list):
        attention_coeff = mean_att[:, c:c + n_atoms]
        mol = Chem.MolFromSmiles(df['smiles'].iloc[j])
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
        for l in range(8):
            drawer = rdMolDraw2D.MolDraw2DSVG(600, 600)
            rdDepictor.Compute2DCoords(mol)
            dos = drawer.drawOptions()
            dos.atomHighlightsAreCircles = True
            dos.fillHighlights = True

            color_dict = {}
            rad_dict = {}
            score_dict = {}
            bond_dict = {}

            arr = attention_coeff[l, :]

            mean_arr = 1 / len(attention_coeff[l, :])

            norm_arr = np.abs((0.8) * ((arr - np.min(arr)) / (np.max(arr)) - np.min(arr)))
            norm_arr = norm_arr[0:-1]
            for t, score in enumerate(norm_arr):
                score = float(score)
                color_dict[t] = [(1, 1 - score, 1 - score)]
                rad_dict[t] = 0.3
                score_dict[t] = arr[t]
            bonds_seq = mol.GetBonds()

            for t in range(len(bonds_seq)):
                if (score_dict[bonds_seq[t].GetBeginAtomIdx()] > mean_arr) and (
                        score_dict[bonds_seq[t].GetEndAtomIdx()] > mean_arr):
                    bond_dict[t] = [(0.9, 0.9, 0.9)]


            drawer.DrawMoleculeWithHighlights(mol, '', color_dict, bond_dict, rad_dict, {})

            drawer.FinishDrawing()
            svg = drawer.GetDrawingText().replace('svg:', '')
            with open("attention_results/" + name + "/" + str(j) + "_" + str(l) + "_" + ".svg", 'w') as file:
                file.write(svg)
                file.close()
    print("Attention images are saved in Folder: "+name)

if __name__ == '__main__':
    warnings.warn = warn
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True, help='input data path', type=str)
    parser.add_argument('-o', '--output', required=True, help='out file name', type=str)
    parser.add_argument('-t', '--sample', required=False, default = 30, help='sampling time', type=int)
    parser.add_argument('-c', '--compute', required=False, default='cpu', help='Computing using CPU or GPU', type=str)
    args = parser.parse_args()
    data_path = args.input
    out_name = args.output
    sampling = args.sample
    computing = args.compute

    if computing.lower() == 'cpu':
        print('use CPU')
        device = 'cpu'
    elif computing.lower() == 'gpu':
        print('use GPU')
        device = 'cuda'
    else: 
        print("Argument error. Compute with CPU")
        device ='cpu'

    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()


    
    df = pd.read_csv(data_path)
    test_data = load_data(df, atom_featurizer, bond_featurizer)
    model = load_model("model/model_weights.pth")

    res_df, num_atom_list, mean_att = prediction(model, df, test_data, device, samples=sampling)
    res_df.to_csv("prediction_results/"+out_name+".csv", index=False)
    attention_visulaizer(out_name, df, mean_att, num_atom_list)
