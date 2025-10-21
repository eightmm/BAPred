import os, random, dgl, torch
import pandas as pd
from tqdm import tqdm
from typing import Union
from dgl.dataloading import GraphDataLoader
from bapred.data.data import BAPredDataset
from bapred.model.model import PredictionPKD

import torch
import numpy as np
import random

# --- 재현성을 위한 설정 (아래 부분을 코드 상단에 추가) ---

# 기본 시드 고정
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# CUDA 연산 시 재현성 보장을 위한 설정
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    
    # cuDNN을 결정론적 모드로 설정
    torch.backends.cudnn.deterministic = True
    
    # cuDNN의 benchmark 기능을 비활성화
    # benchmark=True는 가장 빠른 알고리즘을 찾기 위해 여러번 실행해보는데, 
    # 이는 비결정성을 유발할 수 있음
    torch.backends.cudnn.benchmark = False

def inference(
    protein_pdb: str, 
    ligand_file: str, 
    output: str, 
    batch_size: int, 
 
    model_path: str = './weight', 
    device: Union[str, torch.device] = 'cpu'
) -> None:
    dataset = BAPredDataset(protein_pdb=protein_pdb, ligand_file=ligand_file)
    loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           pin_memory=(device != 'cpu'), num_workers=0)

    model = PredictionPKD(57, 256, 13, 25, 20, 6, 0.2).to(device)
    weight_path = f'{model_path}/BAPred.pth'
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()

    results = {
        "Name": [],
        "pKd": [],
        "Kcal/mol": [],
    }

    with torch.no_grad():
        progress_bar = tqdm(total=len(loader.dataset), unit='ligand')

        for data in loader:
            bgp, bgl, bgc, error, idx, name = data
            bgp, bgl, bgc = bgp.to(device), bgl.to(device), bgc.to(device)

            pkd = model(bgp, bgl, bgc)
            pkd = pkd.view(-1)
            pkd[error == 1] = torch.tensor(float('nan'))

            results["Name"].extend([str(i) for i in name])
            results['pKd'].extend(pkd.tolist())
            results['Kcal/mol'].extend((pkd / -0.73349).tolist())

            progress_bar.update(len(idx))

        progress_bar.close()

    df = pd.DataFrame(results)
    df = df.round(4)
    df.to_csv(output, sep='\t', na_rep='NaN', index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--protein_pdb', default='./example/1KLT_rec.pdb', help='receptor .pdb')
    parser.add_argument('-l', '--ligand_file', default='./example/chk.sdf', help='ligand .sdf/.mol2/.txt')
    parser.add_argument('-o', '--output', default='./example/result.csv', help='result output file')

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

    inference(
        protein_pdb=args.protein_pdb,
        ligand_file=args.ligand_file,
        output=args.output,
        batch_size=args.batch_size,
        ncpu=args.ncpu,
        model_path=args.model_path,
        device=args.device
    )

