import os, random, dgl, torch

import pandas as pd

from tqdm import tqdm

from dgl.dataloading import GraphDataLoader

from data.data import BAPredDataset
from model.model import PredictionPKD


os.environ['PATH'] = '/usr/local/cuda-11.7/bin:' + os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.7/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')


def inference(model, loader, output, device='cpu'):
    model.eval()

    results = {
        "Name": [],
        "pKd": [],
        "Kcal/mol": [],
    }

    with torch.no_grad():
        progress_bar = tqdm( total=len(loader.dataset), unit='ligand' )

        for data in loader:
            bgp, bgl, bgc, error, idx, name = data
            bgp, bgl, bgc = bgp.to(device), bgl.to(device), bgc.to(device)

            pkd = model(bgp, bgl, bgc)

            pkd = pkd.view(-1)

            pkd[error==1] = torch.tensor(float('nan'))

            results["Name"].extend( [ str(i) for i in name ] )
            results['pKd'].extend( pkd.tolist() )
            results['Kcal/mol'].extend( (pkd / -0.73349 ).tolist() )

            progress_bar.update(len(idx))

        progress_bar.close()

    df = pd.DataFrame( results )
    df = df.round(4)
    df.to_csv(output, sep='\t', na_rep='NaN', index=False)


def pkd_prediction( protein_pdb, ligand_sdf, output, batch_size, model_path, device ):
    dataset = BAPredDataset(
        protein_pdb=protein_pdb,
        ligand_sdf=ligand_sdf
    )

    loader = GraphDataLoader( dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = PredictionPKD(57, 256, 13, 25, 20, 6, 0.2).to(device)

    weight_path = f'{model_path}/BAPred.pth'

    model.load_state_dict( torch.load( weight_path )['model_state_dict'] )

    inference( model, loader, output, device )


if __name__ == "__main__" :
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--protein_pdb', default='./example/1KLT_rec.pdb', help='receptor .pdb')
    parser.add_argument('-l','--ligand_sdf', default='./example/chk.sdf', help='ligand .sdf')
    parser.add_argument('-o','--output', default='./example/result.csv', help='result output file')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--ncpu', default=4, type=int, help="cpu worker number")
    parser.add_argument('--device', type=str, default='cuda', help='choose device: cpu or cuda')
    parser.add_argument('--model_path', type=str, default='./weight', help='model weight path')

    args = parser.parse_args()

    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("gpu is not available, run on cpu")
            device = torch.device("cpu")

    pkd_prediction( 
        protein_pdb=args.protein_pdb, 
        ligand_sdf=args.ligand_sdf,
        output=args.output,
        batch_size=args.batch_size,
        model_path=args.model_path,
        device=args.device
    )