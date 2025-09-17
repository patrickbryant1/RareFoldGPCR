import argparse
import sys
import os
import numpy as np
import pandas as pd
import glob
from collections import defaultdict, Counter
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
import gzip
import pdb

parser = argparse.ArgumentParser(description = '''Parse cif file, read target protein chain and write to out in pdb and fasta.
                                                ''')

parser.add_argument('--structure_dir', nargs=1, type= str, default=sys.stdin, help = 'Path to PDB files.')
parser.add_argument('--meta', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with data info/meta.')
parser.add_argument('--line_number', nargs=1, type= int, default=sys.stdin, help = 'What line to use in meta.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

##############FUNCTIONS##############
def read_pdb(pdbname):
    '''Read PDB
    '''

    three_to_one = {'ARG':'R', 'HIS':'H', 'LYS':'K', 'ASP':'D', 'GLU':'E',
                    'SER':'S', 'THR':'T', 'ASN':'N', 'GLN':'Q', 'CYS':'C',
                    'GLY':'G', 'PRO':'P', 'ALA':'A', 'ILE':'I', 'LEU':'L',
                    'MET':'M', 'PHE':'F', 'TRP':'W', 'TYR':'Y', 'VAL':'V',
                    'UNK': 'X'}

    f=gzip.open(pdbname,'rt')


    if '.pdb' in pdbname:
        parser = PDBParser()
        struc = parser.get_structure('', f)
    else:
        parser = MMCIFParser()
        struc = parser.get_structure('',f)

    #Save
    model_coords = {}
    model_seqs = {}
    model_3seq = {}
    model_u3seq = {}
    model_resnos = {}
    model_atoms = {}
    model_bfactors = {}
    model_occupancy = {}
    chain_types = {}


    for model in struc:
        for chain in model:
            #Check if alt conf/str repr.
            if chain.id in [*model_coords.keys()]:
                continue

            #Save
            model_coords[chain.id]=[]
            model_seqs[chain.id]=''
            model_3seq[chain.id]=[]
            model_u3seq[chain.id]=[]
            model_resnos[chain.id]=[]
            model_atoms[chain.id]=[]
            model_bfactors[chain.id]=[]
            model_occupancy[chain.id]=[]

            #Go through all residues
            for residue in chain:
                res_name = residue.get_resname()
                if not is_aa(res_name):
                    continue

                for atom in residue:
                    atom_id = atom.get_id()
                    atm_name = atom.get_name()
                    #Save
                    model_coords[chain.id].append(atom.get_coord())
                    model_3seq[chain.id].append(res_name)
                    model_resnos[chain.id].append(residue.get_id()[1])
                    model_atoms[chain.id].append(atom_id)
                    model_bfactors[chain.id].append(atom.bfactor)
                    model_occupancy[chain.id].append(atom.occupancy)

                #Save residue
                model_seqs[chain.id]+=three_to_one.get(res_name, 'X')
                model_u3seq[chain.id].append(res_name)

            #Count the 'X' - if too many = not protein
            counts = Counter(model_seqs[chain.id])
            if 'X' in [*counts.keys()]:
                frac_X = counts['X']/len(model_seqs[chain.id])
                if frac_X>0.8:
                    chain_types[chain.id] = 'NA'
                else:
                    chain_types[chain.id] = 'Protein'
            else:
                if len(model_seqs[chain.id])>0:
                    chain_types[chain.id] = 'Protein'
                else:
                    chain_types[chain.id] = 'NA'


        return model_coords, model_seqs, model_3seq, model_u3seq, model_resnos, model_atoms, model_bfactors, model_occupancy, chain_types

def format_line(atm_no, atm_name, res_name, chain, res_no, coord, occ, B , atm_id):
    '''Format the line into PDB
    '''

    #Get blanks
    atm_no = ' '*(5-len(atm_no))+atm_no
    atm_name = atm_name+' '*(4-len(atm_name))
    res_name = ' '*(3-len(res_name))+res_name
    res_no = ' '*(4-len(res_no))+res_no
    x,y,z = coord
    x,y,z = str(np.round(x,3)), str(np.round(y,3)), str(np.round(z,3))
    x =' '*(8-len(x))+x
    y =' '*(8-len(y))+y
    z =' '*(8-len(z))+z
    occ = ' '*(6-len(occ))+occ
    B = ' '*(6-len(B))+B

    line = 'ATOM  '+atm_no+'  '+atm_name+res_name+' '+chain+res_no+' '*4+x+y+z+occ+B+' '*11+atm_id+'  '
    return line

def write_pdb(coords, seq, resnos, atoms, bfacs, occupancy, outdir, pdbid):
    """Write PDB
    """


    #Open file
    outname = outdir+pdbid+'.pdb'
    atm_no=0
    with open(outname, 'w') as file:
        for i in range(len(coords)):
            atm_no+=1
            file.write(format_line(str(atm_no), atoms[i], seq[i], 'A', str(resnos[i]), coords[i],str(occupancy[i]),str(bfacs[i]), atoms[i][0])+'\n')




def write_fasta(seq, outdir, pdbid, seqlen):
    """Write fasta
    """
    outname = outdir+pdbid+'.fasta'
    with open(outname, 'w') as file:
        file.write('>'+pdbid+'|'+str(seqlen)+'\n')
        file.write(seq)
##################MAIN#######################

#Parse args
args = parser.parse_args()
#Data
structure_dir = args.structure_dir[0]
meta = pd.read_csv(args.meta[0])
line_number = args.line_number[0]
outdir = args.outdir[0]

#Read structure
row = meta.loc[line_number]
model_coords, model_seqs, model_3seq, model_u3seq, model_resnos, model_atoms, model_bfactors, model_occupancy, chain_types = read_pdb(structure_dir+row.PDB+'.cif.gz')
chain = row['Preferred chain']

#Write protein chain
outdir = outdir+row.PDB+'/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
write_pdb(model_coords[chain], model_3seq[chain], model_resnos[chain], model_atoms[chain], model_bfactors[chain], model_occupancy[chain], outdir, row.PDB+'_'+chain)
seqlen = len(model_seqs[chain])
write_fasta(model_seqs[chain], outdir, row.PDB+'_'+chain, seqlen)
print('Saved to', outdir)
