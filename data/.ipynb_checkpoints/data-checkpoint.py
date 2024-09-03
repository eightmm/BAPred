import torch, dgl
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

from dgl.data import DGLDataset

from .atom_feature import *


def get_ligand_coordinate(mol):
    return mol.GetConformers()[0].GetPositions()

def calculate_pair_distance(arr1, arr2):
    return torch.linalg.norm( arr1[:, None, :] - arr2[None, :, :], axis = -1)

class BAPredDataset(DGLDataset):
    def __init__(self, protein_pdb, ligand_sdf, train=True):
        super(BAPredDataset, self).__init__(name='Protein Ligand Binding Affinity prediction')

        self.ligand_mols = Chem.SDMolSupplier( ligand_sdf )
        self.prot_atom_line, self.prot_atom_coord = self.get_protein_info( protein_pdb )

    def __getitem__(self, idx):
        lmol = self.ligand_mols[idx]
        pmol = self.get_pocket_with_ligand_in_protein( self.prot_atom_line, self.prot_atom_coord, lmol )

        try:
            gl = self.mol_to_graph( lmol )
            gp = self.mol_to_graph( pmol )
            gc = self.complex_to_graph( pmol, lmol )
            error = 0

        except Exception as E:
            print(E)
            gl = self.lig_dummy_graph( num_nodes=2 )
            error = 1

        return gp, gl, gc, error, idx

    def __len__(self):
        return len(self.ligand_mols)

    def lig_dummy_graph(self, num_nodes):
        src = torch.randint(0, num_nodes, (10,))
        dst = torch.randint(0, num_nodes, (10,))
        gl = dgl.graph( (src, dst), num_nodes=num_nodes)
        gl.ndata['feats'] = torch.zeros((num_nodes, 57)).float()  # Example: adding dummy node features
        gl.ndata['pos_enc'] = torch.zeros((num_nodes, 20)).float()  # Example: adding dummy node features
        gl.ndata['coord'] = torch.randn((num_nodes, 3)).float()  # Example: adding dummy node features
        gl.edata['feats'] = torch.zeros((10, 13)).float()
        return gl

    def get_protein_info( self, prot_pdb ):
        prot_atom_line = []
        prot_atom_coord = []
        for line in open(prot_pdb).readlines():
            if line[0:4] in ['ATOM', 'HETA'] and 'H' not in line[12:14] and 'HOH' not in line[17:20]:
                prot_atom_line.append( line )
                prot_atom_coord.append( [ float(line[30:38]), float(line[38:46]), float(line[46:54]) ])

        return prot_atom_line, prot_atom_coord

    def get_pocket_with_ligand_in_protein(self, prot_atom_line, prot_atom_coord, lig_mol ):
        lig_atom_coord = torch.tensor( lig_mol.GetConformers()[0].GetPositions() ).float()
        prot_atom_coord = torch.tensor( prot_atom_coord ).float()

        pl_distance = torch.cdist( prot_atom_coord, lig_atom_coord )

        select_index = torch.where( pl_distance < 8 )[0]
        select_atom = [ line for idx, line in enumerate( prot_atom_line ) if idx in select_index ]

        select_residue = defaultdict(set)
        for idx, line in enumerate(prot_atom_line):
            if idx in select_index:
                select_residue[line[21]].add( int(line[22:26]) )

        total_lines = """"""
        for idx, line in enumerate(prot_atom_line):
            if int( line[22:26] ) in select_residue[ line[21] ]:
                total_lines += line

        mol = Chem.MolFromPDBBlock( total_lines )
        Chem.AssignAtomChiralTagsFromStructure(mol)

        return mol

    def mol_to_graph( self, mol ):
        n     = mol.GetNumAtoms()
        coord = get_mol_coordinate(mol)
        h     = get_atom_feature(mol)
        adj   = get_bond_feature(mol).to_sparse(sparse_dim=2)

        u = adj.indices()[0]
        v = adj.indices()[1]
        e = adj.values()

        g = dgl.DGLGraph()
        g.add_nodes(n)
        g.add_edges(u, v)

        g.ndata['feats'] = h
        g.ndata['coord'] = coord
        g.edata['feats'] = e

        g.ndata['pos_enc'] = dgl.random_walk_pe(g, 20)

        return g

    def complex_to_graph( self, pmol, lmol):
        pcoord = get_mol_coordinate(pmol)
        lcoord = get_mol_coordinate(lmol)
        ccoord = torch.cat( [pcoord, lcoord] )

        npa = pmol.GetNumAtoms()
        nla = lmol.GetNumAtoms()

        distance = calculate_pair_distance(pcoord, lcoord)
        u, v = torch.where( distance < 5 ) ### u - src protein node, v - dst ligand node

        distance = distance[ u, v ].unsqueeze(-1)

        interact_feature = get_interact_feature( pmol, lmol, u, v  )
        distance_feature = get_distance_feature(distance).squeeze(-1)

        e = torch.cat( [interact_feature, distance_feature], dim=1)
        e = torch.cat( [e, e] )

        distance = torch.cat( [ distance, distance] )

        u, v = torch.cat( [u, v+npa] ), torch.cat( [v+npa, u] )

        g = dgl.DGLGraph()
        g.add_nodes( npa + nla )
        g.add_edges( u, v )

        g.ndata['coord'] = ccoord
        g.edata['feats'] = e
        g.edata['distance'] = distance

        return g
