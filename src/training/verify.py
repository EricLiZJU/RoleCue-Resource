"""Verify saved file hashes and reproduce every held-out prediction from checkpoints."""
import argparse
import json
import math
from pathlib import Path
import torch
from src.grammars.generator import sha256, save_json
from src.models import build_model
from src.training.datamodule import read_records
from src.training.eval import predict, score, prediction_rows


def verify(root, output):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    for name, info in manifest['files'].items():
        p = root/name
        if not p.is_file() or sha256(p) != info['sha256'] or p.stat().st_size != info['bytes']:
            raise ValueError(f'Integrity failure: {name}')
    summary = json.loads((root/'summary.json').read_text())
    config = json.loads((root/'config.json').read_text())
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(config['deterministic_algorithms'])
    device = config['device']
    verified = []
    max_error = 0.0
    for run in summary['runs']:
        folder=root/'runs'/run['run_id']
        ckpt=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        if (folder/'last.pt').exists():
            last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
            if last['selection'].startswith('minimum dev') or last['epoch'] != config['training']['epochs']:
                raise ValueError(f'Invalid final checkpoint metadata: {run["run_id"]}')
        elif config.get('require_last_checkpoint'):
            raise ValueError(f'Missing required final checkpoint: {run["run_id"]}')
        model=build_model(ckpt['architecture'],len(ckpt['vocabulary']),ckpt['model_config']).to(device)
        model.load_state_dict(ckpt['model_state'])
        data_root=root/'boundary_data' if run['stage']=='boundary' else Path(config['dataset'])/config['stage']
        for split in (('test','counterfactual') if 'counterfactual' in run else ('test',)):
            rows=read_records(data_root/run['condition']/f'{split}.jsonl','test')
            logits=predict(model,rows,ckpt['vocabulary'],device)
            regenerated=list(prediction_rows(logits,rows))
            saved=read_records(folder/f'{split}_predictions.jsonl')
            if len(saved)!=len(regenerated):
                raise ValueError('Prediction count mismatch')
            error=max(abs(a['p_first_agent']-b['p_first_agent']) for a,b in zip(saved,regenerated))
            max_error=max(max_error,error)
            if error>1e-6 or any(a['record_id']!=b['record_id'] for a,b in zip(saved,regenerated)):
                raise ValueError(f'Checkpoint prediction mismatch: {run["run_id"]}/{split}: {error}')
            current=score(logits,rows)
            if abs(current['nll']-run[split]['nll'])>1e-6 or current['accuracy']!=run[split]['accuracy']:
                raise ValueError(f'Evaluation mismatch: {run["run_id"]}/{split}')
            verified.append(f'{run["run_id"]}/{split}')
        print('VERIFIED',run['run_id'],flush=True)
    result={'status':'passed','manifest_sha256':sha256(root/'manifest.json'),
            'files_verified':len(manifest['files']),'prediction_sets_verified':len(verified),
            'max_probability_error':max_error,'sets':verified}
    save_json(Path(output),result)
    print(json.dumps({k:v for k,v in result.items() if k!='sets'},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--output',default='results/aggregate/training_smoke_v1/verification.json')
    args=parser.parse_args()
    verify(args.root,args.output)
