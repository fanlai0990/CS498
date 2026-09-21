"""Score public CPU tests: python grade_public.py --question 4 --submission ."""
import argparse
from datetime import timedelta
import importlib.util
import itertools
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

# Each case contributes 10% of each component's points. There are three
# public cases and seven TA-only cases per question.
COMPONENT_POINTS = {
    2: {'server': 5, 'worker': 5},
    3: {'allreduce': 10, 'reduce_scatter': 10, 'all_gather': 10},
    4: {'shard': 5, 'sync': 6, 'forward': 4, 'backward': 10},
}
HERE = Path(__file__).resolve().parent

# Public inputs are bundled here; private inputs remain in the TA directory.
PUBLIC_CASES = [{'id': 'q2_public_01',
  'question': 2,
  'visibility': 'public',
  'world': 3,
  'seed': 498,
  'shapes': [[3]],
  'steps': 1,
  'optimizer': 'sgd'},
 {'id': 'q2_public_02',
  'question': 2,
  'visibility': 'public',
  'world': 3,
  'seed': 499,
  'shapes': [[2, 3], [2]],
  'steps': 2,
  'optimizer': 'sgd'},
 {'id': 'q2_public_03',
  'question': 2,
  'visibility': 'public',
  'world': 3,
  'seed': 500,
  'shapes': [[3, 2], [1]],
  'steps': 2,
  'optimizer': 'adam'},
 {'id': 'q3_public_01',
  'question': 3,
  'visibility': 'public',
  'world': 3,
  'seed': 513,
  'shape': [9],
  'steps': 1},
 {'id': 'q3_public_02',
  'question': 3,
  'visibility': 'public',
  'world': 3,
  'seed': 514,
  'shape': [4],
  'steps': 2},
 {'id': 'q3_public_03',
  'question': 3,
  'visibility': 'public',
  'world': 3,
  'seed': 515,
  'shape': [2, 3],
  'steps': 2},
 {'id': 'q4_public_01',
  'question': 4,
  'visibility': 'public',
  'world': 3,
  'seed': 504,
  'shape': [1, 7, 9]},
 {'id': 'q4_public_02',
  'question': 4,
  'visibility': 'public',
  'world': 3,
  'seed': 505,
  'shape': [5, 9, 7]},
 {'id': 'q4_public_03',
  'question': 4,
  'visibility': 'public',
  'world': 3,
  'seed': 506,
  'shape': [11, 6, 3]}]


# ---- Test assertions run inside each distributed worker ----

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def matches(actual, expected):
    try:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        return True
    except (AssertionError, TypeError):
        return False


def all_status(values, world):
    local = torch.tensor(values, dtype=torch.int32)
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    return torch.stack(gathered).bool()


def q2_checks(module, config, rank, world):
    generator = torch.Generator().manual_seed(config['seed'])
    weights = [torch.randn(*shape, generator=generator) for shape in config['shapes']]
    model = torch.nn.ParameterList([torch.nn.Parameter(w.clone()) for w in weights])
    reference = torch.nn.ParameterList([torch.nn.Parameter(w.clone()) for w in weights])
    optimizer_type = torch.optim.Adam if config.get('optimizer') == 'adam' else torch.optim.SGD
    kwargs = {'lr': config.get('lr', 0.03)}
    if optimizer_type is torch.optim.SGD:
        kwargs['momentum'] = config.get('momentum', 0.0)
    opt, ref_opt = optimizer_type(model.parameters(), **kwargs), optimizer_type(reference.parameters(), **kwargs)
    calls = 0
    original_step = opt.step
    def step(*args, **kw):
        nonlocal calls
        calls += 1
        return original_step(*args, **kw)
    opt.step = step
    success = True
    for iteration in range(config.get('steps', 1)):
        for i, (actual, expected) in enumerate(zip(model, reference)):
            if i in config.get('no_grad', []):
                actual.grad = expected.grad = None
                continue
            gradients = [torch.randn(actual.shape, generator=generator) * (r + 1) + (iteration - r) * 0.2 for r in range(world)]
            if config.get('pattern') == 'zero':
                gradients = [torch.zeros_like(g) for g in gradients]
            elif config.get('pattern') == 'cancel':
                gradients[-1] = -sum(gradients[:-1])
            actual.grad = gradients[rank].clone()
            expected.grad = sum(gradients) / world
        module.PS_grads_(model, world_size=world, rankid=rank, opt=opt)
        ref_opt.step()
        success = success and all(matches(a, b) for a, b in zip(model, reference))
        success = success and calls == (iteration + 1 if rank == 0 else 0)
    statuses = all_status([success], world).flatten()
    return {'server': bool(statuses[0]), 'worker': bool(statuses[1:].all())}


def q3_checks(module, config, rank, world):
    original_reduce, original_gather = module.reduce_scatter, module.all_gather
    phases = {'reduce_scatter': [], 'all_gather': []}
    expected_chunks = None
    def reduce_wrapper(*args, **kwargs):
        nonlocal expected_chunks
        chunks = args[0] if args else kwargs['chunks']
        expected_chunks = torch.stack(list(chunks)).clone()
        dist.all_reduce(expected_chunks, op=dist.ReduceOp.SUM)  # Reference only.
        result = original_reduce(*args, **kwargs)
        mask = torch.tensor([matches(chunk, expected) for chunk, expected in zip(chunks, expected_chunks)], dtype=torch.int32)
        masks = [torch.empty_like(mask) for _ in range(world)]
        dist.all_gather(masks, mask)
        # Any consistent rotation of chunk ownership is accepted.
        correct = any(all(masks[r][order[r]].item() for r in range(world)) for order in itertools.permutations(range(world)))
        phases['reduce_scatter'].append(correct)
        return result
    def gather_wrapper(*args, **kwargs):
        chunks = args[0] if args else kwargs['chunks']
        result = original_gather(*args, **kwargs)
        correct = expected_chunks is not None and matches(torch.stack(list(chunks)), expected_chunks)
        phases['all_gather'].append(correct)
        return result
    module.reduce_scatter, module.all_gather = reduce_wrapper, gather_wrapper
    success = True
    generator = torch.Generator().manual_seed(config['seed'])
    for iteration in range(config.get('steps', 1)):
        tensors = [torch.randn(*config['shape'], generator=generator) * (r + 1) + iteration * 0.3 for r in range(world)]
        if config.get('pattern') == 'cancel':
            tensors[-1] = -sum(tensors[:-1])
        elif config.get('pattern') == 'zero':
            tensors = [torch.zeros_like(t) for t in tensors]
        actual, expected = tensors[rank].clone(), sum(tensors) / world
        module.ring_allreduce_(actual, world_size=world, rankid=rank)
        success = success and matches(actual, expected)
    # A one-rank fast path legitimately bypasses the two ring phases.
    values = [success] + [all(phases[name]) and (bool(phases[name]) or world == 1) for name in ('reduce_scatter', 'all_gather')]
    values = all_status(values, world).all(dim=0).tolist()
    return dict(zip(('allreduce', 'reduce_scatter', 'all_gather'), values))


def q4_checks(module, config, rank, world):
    runner = load_module('trusted_q4_runner', Path(__file__).with_name('run.py'))
    runner.q4 = module
    results = {}
    shape = tuple(config['shape'])
    success = runner.component_checks('all', config['seed'], rank, world, cases=[shape], results=results)
    if success and not runner.combined_checks(config['seed'], rank, world, shape=shape):
        results['forward'] = results['backward'] = False
    return results


def worker_main():
    global torch, dist
    import torch
    import torch.distributed as dist
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.case.read_text())
    question = int(config['question'])
    module = load_module(f'q{question}', args.submission.resolve() / f'q{question}.py')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=20))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        functions = {2: q2_checks, 3: q3_checks, 4: q4_checks}
        result = functions[question](module, config, rank, world)
        if rank == 0:
            # Worker logs share stdout and can interleave at arbitrary bytes.
            # Keep machine-readable results separate from diagnostic output.
            args.case.with_suffix('.result.json').write_text(json.dumps(result))
            print('GRADE_RESULT=' + json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()



# ---- Case launching, scoring, and the student command ----

def descendants(parent):
    """Snapshot Linux child PIDs, including torchrun's separate worker groups."""
    parents = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                parents[int(entry.name)] = int(fields[1])
            except (OSError, ValueError, IndexError):
                pass
    found = {parent}
    while True:
        expanded = found | {pid for pid, ppid in parents.items() if ppid in found}
        if expanded == found:
            return found - {parent}
        found = expanded


def stop_case(process):
    # Let torchrun shut its workers down first. Some workers have independent
    # process groups, so killing only the launcher's group can leave pipes open.
    children = descendants(process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        output, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = process.communicate(timeout=5)
    return output


def run_case(case, submission, logs, timeout):
    expected = COMPONENT_POINTS[int(case['question'])]
    with tempfile.TemporaryDirectory(prefix='cs498-grade-') as folder:
        config = Path(folder) / 'case.json'
        config.write_text(json.dumps(case))
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f"--nproc-per-node={case['world']}", str(HERE / 'grade_public.py'), '--worker',
                   '--submission', str(submission), '--case', str(config)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
        timed_out = False
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            output = stop_case(process)
        log_path = logs / (case['id'] + '.log')
        result = {}
        if process.returncode == 0 and not timed_out:
            try:
                result = json.loads(config.with_suffix('.result.json').read_text())
                if not isinstance(result, dict) or any(type(result.get(key)) is not bool for key in expected):
                    raise ValueError('Result must contain a boolean for each component')
            except (OSError, ValueError) as error:
                result = {}
                output += f'\nGRADER ERROR: missing or invalid worker result: {error}\n'
        log_path.write_text(output)
        earned = {key: points / 10 if result.get(key) is True else 0.0
                  for key, points in expected.items()}
        return {'id': case['id'], 'question': case['question'], 'visibility': case['visibility'],
                'earned': round(sum(earned.values()), 6), 'possible': sum(expected.values()) / 10,
                'components': earned, 'timed_out': timed_out, 'exit_code': process.returncode,
                'log': str(log_path)}


def grade(cases, submission, output, timeout=60, jobs=1):
    submission = submission.resolve()
    for question in {int(case['question']) for case in cases}:
        if not (submission / f'q{question}.py').is_file():
            raise FileNotFoundError(submission / f'q{question}.py')
    if timeout <= 0 or not 1 <= jobs <= 4:
        raise ValueError('timeout must be positive and jobs must be between 1 and 4')
    output = output.resolve()
    logs = output.with_suffix('.logs')
    logs.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda case: run_case(case, submission, logs, timeout), cases))
    totals = {}
    for question in sorted({int(c['question']) for c in cases}):
        selected = [r for r in results if int(r['question']) == question]
        totals[str(question)] = {}
        for visibility in ('public', 'hidden'):
            subset = [r for r in selected if r['visibility'] == visibility]
            if subset:
                totals[str(question)][visibility] = {
                    'earned': round(sum(r['earned'] for r in subset), 6),
                    'possible': round(sum(r['possible'] for r in subset), 6)}
        total = round(sum(r['earned'] for r in selected), 6)
        maximum = round(sum(r['possible'] for r in selected), 6)
        print(f'Q{question}: {total:g}/{maximum:g}', flush=True)
    report = {'submission': str(submission), 'totals': totals, 'cases': results}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Report: {output}\nLogs: {logs}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', choices=['2', '3', '4', 'all'], default='all')
    parser.add_argument('--submission', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, default=Path('public_results.json'))
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    cases = PUBLIC_CASES
    if args.question != 'all':
        cases = [case for case in cases if str(case['question']) == args.question]
    grade(cases, args.submission, args.output, args.timeout, args.jobs)


if __name__ == '__main__':
    if '--worker' in sys.argv[1:]:
        worker_main()
    else:
        main()
