"""Dev-selected supervised training; no test data argument or test-based decisions."""
import copy
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from src.models import build_model, parameter_count
from src.training.datamodule import RoleDataset, make_loader, require_disjoint
from src.training.eval import predict, score
from src.grammars.generator import save_json


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def train_one(architecture, train_rows, dev_rows, vocabulary, model_config, settings, output,
              device, seed, progress=None):
    """Optimize only train loss; choose checkpoint only by dev NLL."""
    require_disjoint(train_rows, dev_rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(seed)
    model = build_model(architecture, len(vocabulary), model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                  weight_decay=settings['weight_decay'])
    loader = make_loader(RoleDataset(train_rows, vocabulary), settings['batch_size'], True, seed)
    best_nll, best_epoch, best_state, best_metrics = float('inf'), None, None, None
    started = time.monotonic()
    history, early_history = [], []
    seen = seen_tokens = updates = 0
    early_thresholds = sorted(set(settings.get('early_eval_seen_examples', [])))
    if any(x <= 0 for x in early_thresholds):
        raise ValueError('Early evaluation thresholds must be positive')
    next_early = 0
    initial = score(predict(model, dev_rows, vocabulary, device), dev_rows)
    save_json(output / 'initial_dev.json', initial)
    for epoch in range(1, settings['epochs']+1):
        model.train()
        running_loss = running_correct = count = 0
        epoch_started = time.monotonic()
        for step, (token_ids, target, lengths) in enumerate(loader, 1):
            token_ids, target, lengths = token_ids.to(device), target.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids, lengths)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'])
            if not torch.isfinite(grad_norm):
                raise ValueError('Non-finite gradients')
            optimizer.step()
            batch = len(target)
            running_loss += loss.item()*batch
            running_correct += ((logits.detach() >= 0) == target.bool()).sum().item()
            count += batch
            seen += batch
            seen_tokens += int(lengths.sum().item())
            updates += 1
            while next_early < len(early_thresholds) and seen >= early_thresholds[next_early]:
                dev_early = score(predict(model, dev_rows, vocabulary, device), dev_rows)
                model.train()
                early_row = {'requested_seen_examples': early_thresholds[next_early],
                             'actual_seen_examples': seen, 'seen_tokens': seen_tokens,
                             'optimizer_steps': updates, 'epoch': epoch, 'step': step,
                             'dev': dev_early, 'used_for_selection': False}
                early_history.append(early_row)
                with (output / 'early_learning_curve.jsonl').open('a') as handle:
                    handle.write(json.dumps(early_row, allow_nan=False)+'\n')
                next_early += 1
            if step % settings['log_every_steps'] == 0 or step == len(loader):
                message = (f'{architecture} epoch {epoch:02d}/{settings["epochs"]} '
                           f'step {step:02d}/{len(loader)} train_loss={running_loss/count:.4f} '
                           f'train_acc={running_correct/count:.3f}')
                if progress:
                    progress(message, epoch=epoch, step=step, steps_per_epoch=len(loader))
                else:
                    print(message, flush=True)
        dev = score(predict(model, dev_rows, vocabulary, device), dev_rows)
        if dev['nll'] < best_nll:
            best_nll, best_epoch, best_metrics = dev['nll'], epoch, dev
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # Persist improvements immediately so a long run retains its best model on interruption.
            checkpoint = {'model_state': best_state, 'architecture': architecture, 'model_config': model_config,
                          'vocabulary': vocabulary, 'seed': seed, 'epoch': best_epoch,
                          'selection': 'minimum dev NLL', 'settings': settings}
            temporary = output / 'best.tmp.pt'
            torch.save(checkpoint, temporary)
            temporary.replace(output / 'best.pt')
        row = {'epoch': epoch, 'train_nll': running_loss/count, 'train_accuracy': running_correct/count,
               'dev': dev, 'seen_examples': seen, 'seen_tokens': seen_tokens, 'optimizer_steps': updates,
               'seconds': time.monotonic()-epoch_started, 'best_epoch': best_epoch}
        history.append(row)
        with (output / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
        message = (f'{architecture} epoch {epoch:02d} DEV loss={dev["nll"]:.4f} '
                   f'acc={dev["accuracy"]:.3f} oracle_gap={dev["oracle_nll_regret"]:+.4f} '
                   f'best_epoch={best_epoch}')
        if progress:
            progress(message, epoch=epoch, dev=dev, best_epoch=best_epoch)
        else:
            print(message, flush=True)
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    last_checkpoint = {'model_state': last_state, 'architecture': architecture, 'model_config': model_config,
                       'vocabulary': vocabulary, 'seed': seed, 'epoch': settings['epochs'],
                       'selection': 'final training epoch; not used for primary test selection',
                       'settings': settings, 'dev': history[-1]['dev']}
    torch.save(last_checkpoint, output / 'last.pt')
    model.load_state_dict(best_state)
    checkpoint = {'model_state': best_state, 'architecture': architecture, 'model_config': model_config,
                  'vocabulary': vocabulary, 'seed': seed, 'epoch': best_epoch,
                  'selection': 'minimum dev NLL', 'settings': settings}
    torch.save(checkpoint, output / 'best.pt')
    summary = {'architecture': architecture, 'seed': seed, 'parameters': parameter_count(model),
               'initial_dev': initial, 'best_epoch': best_epoch, 'best_dev': best_metrics,
               'epochs_completed': settings['epochs'], 'optimizer_steps': updates,
               'seen_examples': seen, 'seen_tokens': seen_tokens, 'seconds': time.monotonic()-started,
               'device': str(device), 'selection': 'minimum dev NLL', 'history': history,
               'early_learning_curve': early_history,
               'checkpoints': {'best': 'best.pt', 'last': 'last.pt'}}
    save_json(output / 'training.json', summary)
    return model, summary
