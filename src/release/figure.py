"""Reproduce the Core reliability-grid main figure from frozen summaries."""
import hashlib
import json
from pathlib import Path


def reliability_grid(snapshot_dir, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    snapshot_dir, output_dir = Path(snapshot_dir), Path(output_dir)
    source = snapshot_dir/'main_aggregate.json'
    aggregate = json.loads(source.read_text())
    groups = {(row['condition'], row['architecture']): row for row in aggregate['groups']}
    architectures = ('rnn', 'lstm', 'transformer')
    values = {}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'savefig.dpi': 250})
    figure, axes = plt.subplots(1, 3, figsize=(6.5, 2.15), layout='constrained')
    for axis, arch in zip(axes, architectures):
        grid = np.array([[100*groups[f'ro{o:03d}_rm{m:03d}', arch]
                          ['metrics']['conflict_oracle_probability_mae']['mean']
                          for m in (55, 75, 95)] for o in (55, 75, 95)])
        values[arch] = grid.tolist()
        image = axis.imshow(grid, origin='lower', vmin=0, vmax=2.5,
                            cmap='cividis', aspect='equal')
        for i in range(3):
            for j in range(3):
                axis.text(j, i, f'{grid[i,j]:.2f}', ha='center', va='center',
                          color='white' if grid[i,j] < 1.2 else 'black', fontsize=8)
        axis.set(xticks=range(3), xticklabels=['.55', '.75', '.95'],
                 yticks=range(3), yticklabels=['.55', '.75', '.95'],
                 xlabel=r'Marking reliability $r_m$', title=arch.upper())
    axes[0].set_ylabel(r'Order reliability $r_o$')
    figure.colorbar(image, ax=axes, shrink=.85, pad=.02, label='Conflict MAE (pp)')
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        figure.savefig(output_dir/f'reliability_grid.{extension}', bbox_inches='tight', pad_inches=.04)
    plt.close(figure)
    report = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'units': 'posterior mean absolute error in percentage points',
              'architectures': values, 'conditions_order': [.55, .75, .95],
              'conditions_marking': [.55, .75, .95]}
    (output_dir/'reliability_grid_values.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def binding_diagnostic(analysis_file, output_dir):
    """Recreate the Binding seed-level information-gap figure from frozen analysis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    analysis_file, output_dir = Path(analysis_file), Path(output_dir)
    analysis = json.loads(analysis_file.read_text())
    if analysis['status'] != 'completed' or analysis['run_count'] != 45:
        raise ValueError('Expected complete 45-run Binding analysis')
    conditions = ('ro055_rm095', 'ro075_rm075', 'ro095_rm055')
    architectures = ('rnn', 'lstm', 'transformer')
    names = ('Strong marking', 'Equal reliability', 'Strong order')
    colors = {'rnn': '#0072B2', 'lstm': '#D55E00', 'transformer': '#009E73'}
    lookup = {(row['condition'], row['architecture']): row for row in analysis['groups']}
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True, layout='constrained')
    for axis, condition, title in zip(axes, conditions, names):
        baseline = analysis['baselines'][condition]['test']['binding']
        information_gap = baseline['suffix']['nll'] - baseline['full_oracle']['nll']
        axis.axhline(information_gap, color='black', linestyle='--', linewidth=1)
        axis.axhline(0, color='#888888', linewidth=.65)
        for index, architecture in enumerate(architectures):
            row = lookup[condition, architecture]
            for version, offset, marker, alpha in (('binding', -.13, 'o', .9), ('local', .13, 's', .55)):
                values = list(row['test'][version]['metrics']['nll_regret']['training_seed']['seed_values'].values())
                x = index + offset
                for step, value in enumerate(values):
                    axis.scatter(x + (step-2)*.018, value, s=17, color=colors[architecture],
                                 marker=marker, alpha=alpha)
                axis.plot(x, sum(values)/5, marker='_', markersize=12, markeredgewidth=2,
                          color=colors[architecture], linestyle='none')
        axis.set(title=title, xticks=range(3), xticklabels=['RNN', 'LSTM', 'Trf'],
                 ylim=(-.04, .67))
        axis.grid(axis='y', alpha=.16)
    axes[0].set_ylabel('Oracle NLL regret (nats/input)')
    figure.legend(handles=[Line2D([], [], color='black', linestyle='--', label='Suffix information gap'),
                           Line2D([], [], color='#444444', marker='o', linestyle='none', label='Binding seed'),
                           Line2D([], [], color='#888888', marker='s', linestyle='none', label='Local seed')],
                  loc='outside lower center', ncol=3, frameon=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        figure.savefig(output_dir / f'binding_diagnostic.{extension}',
                       bbox_inches='tight', pad_inches=.04, dpi=250)
    plt.close(figure)
    return {'source_sha256': hashlib.sha256(analysis_file.read_bytes()).hexdigest(),
            'run_count': 45, 'conditions': list(conditions)}
