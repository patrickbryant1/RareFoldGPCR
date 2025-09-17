import json
import os
import warnings
import pathlib
import pickle
import random
import sys
#Insert path
#sys.path.insert(0, '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1]))
sys.path.insert(0, '/proj/berzelius-2023-267/users/x_patbr/software/rare_fold/src/net/')
import time
from typing import Dict, Optional
from typing import NamedTuple
import haiku as hk
import jax
import jax.numpy as jnp
import optax
#Silence tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.config.set_visible_devices([], 'GPU')

import argparse
import pandas as pd
import numpy as np
from collections import Counter
from scipy.special import softmax
import copy
from ast import literal_eval

import pdb


#AlphaFold imports - now RareFold
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.model import data
from alphafold.model import config
from alphafold.model import features
from alphafold.model import modules

#JAX will preallocate 90% of currently-available GPU memory when the first JAX operation is run.
#This prevents this
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

parser = argparse.ArgumentParser(description = """Batch fwd pass for evaluating prediction variability of designed sequences.""")


parser.add_argument('--predict_id', nargs=1, type= str, default=sys.stdin, help = 'Id to predict.')
parser.add_argument('--MSA_feats', nargs=1, type= str, default=sys.stdin, help = 'MSA feats for receptor.')
parser.add_argument('--num_recycles', nargs=1, type= int, default=sys.stdin, help = 'Number of recycles.')
parser.add_argument('--binder_sequence', nargs=1, type= str, default=sys.stdin, help = 'Sequence for binder in 3-letter code.')
parser.add_argument('--batch_size', nargs=1, type= int, default=sys.stdin, help = 'Batch size (will run design threads in parallel).')
parser.add_argument('--params', nargs=1, type= str, default=sys.stdin, help = 'Params to use.')
parser.add_argument('--cyclic_offset', nargs=1, type= str, default=sys.stdin, help = 'Use a cyclic offset for the binder (True) or not (False).')
parser.add_argument('--num_clusters', nargs=1, type= int, default=sys.stdin, help = 'Number of MSA clusters to use.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

##############FUNCTIONS##############
##########INPUT DATA#########
def process_features(raw_features, config, random_seed):
    """Processes features to prepare for feeding them into the model.

    Args:
    raw_features: The output of the data pipeline either as a dict of NumPy
      arrays or as a tf.train.Example.
    random_seed: The random seed to use when processing the features.

    Returns:
    A dict of NumPy feature arrays suitable for feeding into the model.
    """
    return features.np_example_to_features(np_example=raw_features,
                                            config=config,
                                            random_seed=random_seed)


def process_input_feats(new_feature_dict, config):
    """
    Load all input feats.
    """


    #Number of possible amino acids
    num_AAs = len(residue_constants.restype_name_to_atom14_names.keys())
    #Max number of atoms per amino acid in the dense representation
    num_dense_atom_max = len(residue_constants.restype_name_to_atom14_names['ALA'])
    #Process the features on CPU (sample MSA)
    #This also creates mappings for the atoms: 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists'
    new_feature_dict['aatype'] =  np.eye(num_AAs)[new_feature_dict['int_seq']]
    processed_feature_dict = process_features(new_feature_dict, config, np.random.choice(sys.maxsize))

    #Cyclic
    if config.model.embeddings_and_evoformer.cyclic_offset==True:
        pos = new_feature_dict['residue_index']
        cyclic_offset_array = pos[:, None] - pos[None, :]
        binder_cyclic_offset_array = new_feature_dict['binder_cyclic_offset_array']
        cyclic_offset_array[-len(binder_cyclic_offset_array):,-len(binder_cyclic_offset_array):]=binder_cyclic_offset_array
        new_feature_dict['cyclic_offset'] = cyclic_offset_array



    #Arrange feats
    batch_ex = copy.deepcopy(new_feature_dict)

    #If Rare amino acids in the receptor - this has to be specified here
    #batch_ex['aatype'] = rare_feats['onehot_seq'] #Use the sequence from the structure here - RARE!!!
    batch_ex['aatype'] = new_feature_dict['int_seq']
    batch_ex['seq_mask'] = processed_feature_dict['seq_mask']
    batch_ex['msa_mask'] = processed_feature_dict['msa_mask']
    batch_ex['residx_atom14_to_atom37'] = processed_feature_dict['residx_atom14_to_atom37']
    batch_ex['residx_atom37_to_atom14'] = processed_feature_dict['residx_atom37_to_atom14']
    batch_ex['atom37_atom_exists'] = processed_feature_dict['atom37_atom_exists']
    batch_ex['extra_msa'] = processed_feature_dict['extra_msa']
    batch_ex['extra_msa_mask'] = processed_feature_dict['extra_msa_mask']
    batch_ex['bert_mask'] = processed_feature_dict['bert_mask']
    batch_ex['true_msa'] = processed_feature_dict['true_msa']
    batch_ex['extra_has_deletion'] = processed_feature_dict['extra_has_deletion']
    batch_ex['extra_deletion_value'] = processed_feature_dict['extra_deletion_value']
    batch_ex['msa_feat'] = processed_feature_dict['msa_feat']

    #Target feats have to be updated with the onehot_seq from the structure to include the modified amino acids
    batch_ex['target_feat'] = np.eye(num_AAs)[new_feature_dict['int_seq']]
    batch_ex['atom14_atom_exists'] = processed_feature_dict['atom14_atom_exists']
    batch_ex['residue_index'] = processed_feature_dict['residue_index']

    return batch_ex


def init_features(feature_dict, onehot_binder_seq, binder_length, config):
    """Update the features to include the binder sequence

    #From MSA feats
    'aatype',
    'between_segment_residues',
    'domain_name',
    'residue_index',
    'seq_length',
    'sequence',
    'deletion_matrix_int',
    'msa',
    'num_alignments'
    """

    #Save
    new_feature_dict = {}

    #Add peptide feats to feature dict
    #aatype
    new_feature_dict['int_seq'] = np.concatenate((np.argmax(feature_dict['aatype'],axis=1), np.array(onehot_binder_seq)),axis=0)
    #between_segment_residues
    new_feature_dict['between_segment_residues'] = np.concatenate((feature_dict['between_segment_residues'],np.zeros((binder_length), dtype=np.int32)),axis=0)
    #residue_index
    new_feature_dict['residue_index'] = np.concatenate((feature_dict['residue_index'],np.array(range(binder_length), dtype=np.int32)+feature_dict['residue_index'][-1]+201), axis=0)
    #seq_length
    new_feature_dict['seq_length'] = np.array([new_feature_dict['int_seq'].shape[0]] * new_feature_dict['int_seq'].shape[0], dtype=np.int32)

    #Merge MSA features
    #deletion_matrix_int
    new_feature_dict['deletion_matrix_int']=np.concatenate((feature_dict['deletion_matrix_int'],
                                            np.zeros((feature_dict['deletion_matrix_int'].shape[0],binder_length))), axis=1)
    #msa
    peptide_msa = np.zeros((feature_dict['msa'].shape[0],binder_length),dtype=int)
    peptide_msa[:,:] = 21
    #Assign first seq - need to have X instead of mod AAs
    """
    HHBLITS_AA_TO_ID = {'A': 0,'B': 2,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'J': 20,'K': 8,'L': 9,'M': 10,'N': 11,
                        'O': 20,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'U': 1,'V': 17,'W': 18,'X': 20,'Y': 19,'Z': 3,'-': 21,}
    """
    x = copy.deepcopy(np.array(onehot_binder_seq))
    x[x>19]=20
    peptide_msa[0,:] = x

    new_feature_dict['msa']=np.concatenate((feature_dict['msa'], peptide_msa), axis=1)

    #num_alignments
    new_feature_dict['num_alignments']=np.concatenate((feature_dict['num_alignments'], feature_dict['num_alignments'][:len(onehot_binder_seq)]), axis=0)

    #Process
    if config.model.embeddings_and_evoformer.cyclic_offset==True:
        new_feature_dict['binder_cyclic_offset_array'] = copy.deepcopy(feature_dict['binder_cyclic_offset_array'])

    new_feature_dict = process_input_feats(new_feature_dict, config)


    return new_feature_dict



##########MODEL and DESIGN#########

def eval_binder(config,
                predict_id,
                MSA_feats,
                num_recycles=3,
                binder_length=10,
                batch_size=1,
                params=None,
                binder_sequence=None,
                outdir=None):
    """Design a binder
    """

    #Initialize AA mapping
    all_AAs = np.array([*residue_constants.restype_name_to_atom14_names.keys()])
    AA_mapping = {}
    for i in range(len(all_AAs)):
        AA_mapping[all_AAs[i]]=i

    int_binder_seq = []
    for res3 in binder_sequence.split('-'):
        int_binder_seq.append(AA_mapping.get(res3,20))

    if config.model.embeddings_and_evoformer.cyclic_offset==True:
        cyclic_offset_array = np.zeros((binder_length, binder_length))
        cyc_row = np.arange(0,-binder_length,-1)
        pc = int(np.round(binder_length/2)) #Get centre
        cyc_row[pc+1:]=np.arange(len(cyc_row[pc+1:]),0,-1)
        for i in range(len(cyclic_offset_array)):
            cyclic_offset_array[i]=np.roll(cyc_row,i)
        MSA_feats['binder_cyclic_offset_array']=cyclic_offset_array

    #Define the forward function
    def _forward_fn(batch):
        '''Define the forward function - has to be a function for JAX
        '''
        model = modules.AlphaFold(config.model)

        return model(batch,
                    is_training=False,
                    compute_loss=False,
                    ensemble_representations=False,
                    return_representations=True)

    #The forward function is here transformed to apply and init functions which
    #can be called during training and initialisation (JAX needs functions)
    forward = hk.transform(_forward_fn)
    #Function to vmap - this is usually wrapped with functools
    #This causes communication errors btw my processes in HPC envs, however
    vmap_apply_fwd = jax.vmap(forward.apply, (None,None,0)) #None over params and rng, but 0 over batch

    #Get a random key
    rng = jax.random.PRNGKey(42)

    #Load params (need to do this here - need to enable GPU through jax first)
    params = np.load(params, allow_pickle=True)


    print('Making feats...')
    #Make feature dicts
    init_feature_dicts = [init_features(MSA_feats, int_binder_seq, binder_length, config) for x in range(batch_size)]
    batch = {}
    for key in init_feature_dicts[0]:
        batch[key] = np.array([init_feature_dicts[x][key] for x in range(batch_size)])
        batch[key] = np.reshape(batch[key], (batch_size, 1, *batch[key].shape[1:]))

    batch['num_iter_recycling'] = np.zeros((batch_size, 1,))
    batch['num_iter_recycling'][:] = num_recycles
    print('Predicting...')
    t0 = time.time()
    prediction_result = vmap_apply_fwd(params, rng, batch)
    print('Prediction took',time.time()-t0,'s')
    #Save all
    t0 = time.time()
    [save_structure(batch, prediction_result, i, predict_id, outdir) for i in range(batch_size)]
    print('Saving preds took',time.time()-t0,'s')


def save_structure(batch, prediction_result, i, pred_id, outdir):
    """Save prediction

    save_feats = {'aatype':batch['aatype'][0][0], 'residue_index':batch['residue_index'][0][0]}
    result = {'predicted_lddt':aux['predicted_lddt'],
            'structure_module':{'final_atom_positions':aux['structure_module']['final_atom_positions'][0],
            'final_atom_mask': aux['structure_module']['final_atom_mask'][0]
            }}
    save_structure(save_feats, result, step_num, outdir)

    """
    #Define the plDDT bins
    bin_width = 1.0 / 50
    bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)

    #for i in range(len(batch['aatype'])):
    #Save structure
    save_feats = {'aatype':batch['aatype'][i], 'residue_index':batch['residue_index'][i]}
    result = {'predicted_lddt':prediction_result['predicted_lddt']['logits'][i],
            'structure_module':{'final_atom_positions':prediction_result['structure_module']['final_atom_positions'][i],
            'final_atom_mask': prediction_result['structure_module']['final_atom_mask'][i]
            }}
    # Add the predicted LDDT in the b-factor column.
    plddt_per_pos = jnp.sum(jax.nn.softmax(result['predicted_lddt']) * bin_centers[None, :], axis=-1)
    plddt_b_factors = np.repeat(plddt_per_pos[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(features=save_feats, result=result,  b_factors=plddt_b_factors)
    unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(outdir+'/', pred_id+'_'+str(i)+'.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
        f.write(unrelaxed_pdb)



##################MAIN#######################

#Parse args
args = parser.parse_args()
predict_id = args.predict_id[0]
MSA_feats = np.load(args.MSA_feats[0], allow_pickle=True)
num_recycles = args.num_recycles[0]
binder_sequence = args.binder_sequence[0]
binder_length = len(binder_sequence.split('-'))
batch_size = args.batch_size[0]
params = args.params[0]
cyclic_offset = args.cyclic_offset[0]
if cyclic_offset=='True':
    cyclic_offset=True
else:
    cyclic_offset=None
num_clusters = args.num_clusters[0]
outdir = args.outdir[0]


#Update config
config.CONFIG.model.embeddings_and_evoformer['cyclic_offset'] = cyclic_offset
config.CONFIG.data.eval.max_msa_clusters = num_clusters

#Predict
eval_binder(config.CONFIG,
            predict_id,
            MSA_feats,
            num_recycles=num_recycles,
            binder_length=binder_length,
            batch_size=batch_size,
            params=params,
            binder_sequence = binder_sequence,
            outdir=outdir)
