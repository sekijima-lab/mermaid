import sys,json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import RWMol
n=json.loads(Path(sys.argv[1]).read_text())
exec(''.join(n['cells'][2]['source']))
s='Cn1cc(C(=O)Nc2ccccc2C(=O)NCCc2ccccc2)c(=O)c2cccn21'
mol=setBFSorder(Chem.MolFromSmiles(s));original=Chem.MolToSmiles(mol)
fragments=[]
for ids in [[25,26,27,28,29,30,31],[1,2,3,4,5,6,7,9,10,11,14]]:
 m=clear_atommap(rm_atom(mol,ids));fragments.append(Chem.MolToSmiles(m));assert '<svg' in DrawMol(m)
print(json.dumps({'original':original,'fragments':fragments}))
