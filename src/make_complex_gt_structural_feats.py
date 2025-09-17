import os
import numpy as np
import argparse
import sys
from Bio.PDB.internal_coords import *
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from rarefold.model import quat_affine
from rarefold.common import residue_constants
import pickle

import pdb

parser = argparse.ArgumentParser(description = """Builds the ground truth structural features for the loss calculations.""")

parser.add_argument('--input_pdb', nargs=1, type= str, default=sys.stdin, help = 'Path to input pdb file.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')


##############FUNCTIONS##############
def read_pdb(pdbname):
    '''Read PDB and format to make structural features
    The name atom14 is still used here as a legacy from AF2, but the shape is now the number of atoms
    in the biggest amino acid (dense representation)
    '''

    #NOTE!
    #Make sure this order corresponds to the one in residue_constants.py
    restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'CB', 'O', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] , #0
    'ARG': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD', 'NE', 'NH1', 'NH2', 'CZ', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'ASN': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'ND2', 'OD1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'ASP': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'OD1', 'OD2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'CYS': ['N', 'CA', 'C', 'CB', 'O', 'SG', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'GLN': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD', 'NE2', 'OE1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] , #5
    'GLU': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'GLY': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'HIS': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD2', 'ND1', 'CE1', 'NE2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'ILE': ['N', 'CA', 'C', 'CB', 'O', 'CG1', 'CG2', 'CD1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'LEU': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] , #10
    'LYS': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'MET': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'SD', 'CE', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'PHE': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'PRO': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'SER': ['N', 'CA', 'C', 'CB', 'O', 'OG', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] , #15
    'THR': ['N', 'CA', 'C', 'CB', 'O', 'CG2', 'OG1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'TRP': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CH2', 'CZ2', 'CZ3', '', '', '', '', '', '', '', '', '', '', ''] ,
    'TYR': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'OH', 'CZ', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'VAL': ['N', 'CA', 'C', 'CB', 'O', 'CG1', 'CG2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'UNK': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] , #index 20
    'MSE': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'SE', 'CE', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'TPO': ['N', 'CA', 'C', 'CB', 'O', 'CG2', 'OG1', 'P', 'O1P', 'O2P', 'O3P', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'MLY': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD', 'CE', 'NZ', 'CH1', 'CH2', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'CME': ['N', 'CA', 'C', 'CB', 'O', 'SG', 'SD', 'CE', 'CZ', 'OH', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'PTR': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'OH', 'CZ', 'P', 'O1P', 'O2P', 'O3P', '', '', '', '', '', '', '', '', ''] ,
    'SEP': ['N', 'CA', 'C', 'CB', 'O', 'OG', 'P', 'O1P', 'O2P', 'O3P', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'SAH': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'SD', "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'] ,
    'CSO': ['N', 'CA', 'C', 'CB', 'O', 'SG', 'OD', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'PCA': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD', 'OE', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'KCX': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD', 'CE', 'NZ', 'CX', 'OQ1', 'OQ2', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'CAS': ['N', 'CA', 'C', 'CB', 'O', 'SG', 'AS', 'CE1', 'CE2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'CSD': ['N', 'CA', 'C', 'CB', 'O', 'SG', 'OD1', 'OD2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'MLZ': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD', 'CE', 'NZ', 'CM', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'OCS': ['N', 'CA', 'C', 'CB', 'O', 'SG', 'OD1', 'OD2', 'OD3', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'ALY': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD', 'CE', 'NZ', 'OH', 'CH', 'CH3', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'CSS': ['N', 'CA', 'C', 'CB', 'O', 'SG', 'SD', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'CSX': ['N', 'CA', 'C', 'CB', 'O', 'SG', 'OD', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'HIC': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD2', 'ND1', 'CE1', 'NE2', 'CZ', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'HYP': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD', 'OD1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'YCM': ['N', 'CA', 'C', 'CB', 'O', 'SG', 'CD', 'CE', 'OZ1', 'NZ2', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'YOF': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'OH', 'CZ', 'F', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'M3L': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD', 'CE', 'NZ', 'CM1', 'CM2', 'CM3', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'PFF': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'F', 'CZ', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'CGU': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD1', 'OE11', 'OE12', 'CD2', 'OE21', 'OE22', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'FTR': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CH2', 'CZ2', 'CZ3', 'F', '', '', '', '', '', '', '', '', '', ''] ,
    'LLP': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CD', 'CE', 'NZ', "C4'", 'C2', 'C3', 'C4', 'C5', 'C6', 'N1', "C2'", 'O3', "C5'", 'OP4', 'P', 'OP1', 'OP2', 'OP3', ''] ,
    'CAF': ['N', 'CA', 'C', 'CB', 'O', 'SG', 'AS', 'CE1', 'CE2', 'O1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'CMH': ['N', 'CA', 'C', 'CB', 'O', 'SG', 'HG', 'CM', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    'MHO': ['N', 'CA', 'C', 'CB', 'O', 'CG', 'SD', 'CE', 'OD1', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] ,
    }

    #List of modified amino acids
    mod_amino_acids = ['MSE', 'TPO', 'MLY', 'CME', 'PTR', 'SEP', 'SAH','CSO', 'PCA', 'KCX',
                        'CAS', 'CSD', 'MLZ', 'OCS', 'ALY', 'CSS', 'CSX', 'HIC', 'HYP', 'YCM',
                        'YOF', 'M3L', 'PFF', 'CGU', 'FTR', 'LLP', 'CAF', 'CMH', 'MHO']

    restype_keys = np.array([*restype_name_to_atom14_names.keys()])

    if '.pdb' in pdbname:
        parser = PDBParser()
        struc = parser.get_structure('', pdbname)
    else:
        parser = MMCIFParser()
        struc = parser.get_structure('',pdbname)

    #Save
    model_coords14 = {} #Atom coords
    model_coords14_mask = {} #What coords are present
    model_atom14_atom_exists = {} #If the atom exists in atom14
    model_resnos = {} #Residue nummbers
    model_chi_angles = {} #Chi angles
    model_chi_angles_mask = {} #Chi angles mask
    model_pseudo_beta = {} #Pseudo beta pos (CA for GLY)
    model_pseudo_beta_mask = {} #If pseudo beta exists
    model_onehot_seq = {} #Onehot seq
    model_mod_aa_index = {} #What amino acids are modified


    #Max number of atoms per amino acid in the dense representation
    num_dense_atom_max = len(residue_constants.restype_name_to_atom14_names['ALA'])


    for model in struc:
        for chain in model:
            # compute bond lengths, angles, dihedral angles
            chain.atom_to_internal_coordinates(verbose=True)
            #Save
            model_coords14[chain.id]=[]
            model_coords14_mask[chain.id]=[]
            model_atom14_atom_exists[chain.id]=[]
            model_resnos[chain.id]=[]
            model_chi_angles[chain.id]=[]
            model_chi_angles_mask[chain.id]=[]
            model_pseudo_beta[chain.id]=[]
            model_pseudo_beta_mask[chain.id]=[]
            model_onehot_seq[chain.id]=[]
            model_mod_aa_index[chain.id]=[]


            #Go through al residues
            for residue in chain:
                #Get res name for atom14 mapping
                res_name = residue.get_resname()
                if not is_aa(res_name):
                    continue
                res_atom14 = np.zeros((num_dense_atom_max,3))
                res_atom14_mask = np.zeros(num_dense_atom_max)
                #Check if the residue is known
                if res_name not in [*restype_name_to_atom14_names.keys()]:
                    res_name = 'UNK'
                #Positions
                res14_pos = np.array(restype_name_to_atom14_names[res_name])
                res_atom14_exists = np.zeros(num_dense_atom_max)
                res_atom14_exists[np.argwhere(res14_pos!='')[:,0]]=1
                #Get chi angles
                res_chi = np.zeros(4)
                res_chi_mask = np.zeros(4)
                chi1 = residue.internal_coord.get_angle("chi1")
                chi2 = residue.internal_coord.get_angle("chi2")
                chi3 = residue.internal_coord.get_angle("chi3")
                chi4 = residue.internal_coord.get_angle("chi4")
                if chi1:
                    res_chi[0]=chi1
                    res_chi_mask[0]=1
                if chi2:
                    res_chi[1]=chi2
                    res_chi_mask[2]=1
                if chi3:
                    res_chi[2]=chi3
                    res_chi_mask[2]=1
                if chi4:
                    res_chi[3]=chi4
                    res_chi_mask[3]=1
                #Get coords
                pb_fetched = False
                for atom in residue:
                    atom_id = atom.get_id()
                    if atom_id=='CB' or (atom_id=='CA' and res_name=='GLY'):
                        model_pseudo_beta[chain.id].append(atom.get_coord())
                        pb_fetched=True
                        model_pseudo_beta_mask[chain.id].append(1)
                    #Save
                    if atom_id in res14_pos:
                        map_pos = np.argwhere(res14_pos==atom_id)[0][0]
                        res_atom14[map_pos,:]=atom.get_coord()
                        res_atom14_mask[map_pos]=1
                #Save residue info
                model_coords14[chain.id].append(res_atom14)
                model_coords14_mask[chain.id].append(res_atom14_mask)
                model_atom14_atom_exists[chain.id].append(res_atom14_exists)
                model_resnos[chain.id].append(residue.get_id()[1])
                model_chi_angles[chain.id].append(res_chi)
                model_chi_angles_mask[chain.id].append(res_chi_mask)
                model_onehot_seq[chain.id].append(np.argwhere(restype_keys==res_name)[0][0])
                if res_name in mod_amino_acids:
                    model_mod_aa_index[chain.id].append(1)
                else:
                    model_mod_aa_index[chain.id].append(0)

                #Check if pseudo beta was fetched
                if pb_fetched==False:
                    model_pseudo_beta[chain.id].append(np.array([0,0,0]))
                    model_pseudo_beta_mask[chain.id].append(0)

    model_feats = {'model_coords14':model_coords14,
                   'model_coords14_mask':model_coords14_mask,
                   'model_atom14_atom_exists':model_atom14_atom_exists,
                   'model_resnos':model_resnos,
                   'model_chi_angles':model_chi_angles,
                   'model_chi_angles_mask':model_chi_angles_mask,
                   'model_pseudo_beta':model_pseudo_beta,
                   'model_pseudo_beta_mask':model_pseudo_beta_mask,
                   'model_onehot_seq':model_onehot_seq,
                   'model_mod_aa_index':model_mod_aa_index,
                   }

    if np.sum(model_mod_aa_index[chain.id])>0:
        print('Mod AA index', np.argwhere(np.array(model_mod_aa_index[chain.id])>0))

    return model_feats

##################MAIN#######################

#Parse args
args = parser.parse_args()
#Data
input_pdb = args.input_pdb[0]
outdir = args.outdir[0]
#Read
model_feats = read_pdb(input_pdb)

#Save
structure_feats = {}
for chain in model_feats['model_coords14']:
    if chain not in [*structure_feats.keys()]:
        structure_feats[chain] = {}
    structure_feats[chain]['atom14_gt_positions'] = np.array(model_feats['model_coords14'][chain])
    structure_feats[chain]['atom14_gt_exists'] = np.array(model_feats['model_coords14_mask'][chain])
    structure_feats[chain]['atom14_atom_exists'] = np.array(model_feats['model_atom14_atom_exists'][chain])
    structure_feats[chain]['residue_index'] = np.array(model_feats['model_resnos'][chain])
    structure_feats[chain]['chi_angles'] = np.array(model_feats['model_chi_angles'][chain])
    structure_feats[chain]['chi_mask'] = np.array(model_feats['model_chi_angles_mask'][chain])
    structure_feats[chain]['pseudo_beta'] = np.array(model_feats['model_pseudo_beta'][chain])
    structure_feats[chain]['pseudo_beta_mask'] = np.array(model_feats['model_pseudo_beta_mask'][chain])
    structure_feats[chain]['onehot_seq'] = np.array(model_feats['model_onehot_seq'][chain])
    structure_feats[chain]['mod_aa_index'] = np.array(model_feats['model_mod_aa_index'][chain])
#Write out features as a pickled dictionary.
features_output_path = os.path.join(outdir, 'complex_structure_features.pkl')
with open(features_output_path, 'wb') as f:
    pickle.dump(structure_feats, f, protocol=4)
print('Saved features to',features_output_path)
