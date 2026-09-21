#!/usr/bin/env python3
"""Diagnostic runner for Q2/Q3 Qwen training and Q4 tensor-parallel checks.

Select --question 2, 3, or 4. Q4 uses only small CPU tensors and PyTorch;
Transformers and Datasets are loaded only for Q2/Q3 training.
"""
from contextlib import contextmanager
from datetime import timedelta
import importlib

import math
import hashlib, datetime as dt
import argparse, os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import time

q4 = None  # Selected lazily by Q4 diagnostics or injected by the trusted grader.

###a example of LLM templete for reference###
###you don't have to actually use it###
class LLMTemplete(nn.Module):                      
    def __init__(self, model, end):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = nn.ModuleList(model.model.layers[:end])     
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary_emb = model.model.rotary_emb               

    def forward(self, input_ids):
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1).contiguous()
        hidden = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        attention_mask = torch.triu(
            torch.full((seqlen, seqlen), float('-inf'), device=device, dtype=hidden.dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                output_attentions=False,
                use_cache=False,
            )
            # Transformers versions may return either a tensor or a tuple.
            hidden = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        hidden = self.norm(hidden)


        return self.lm_head(hidden)


###dataset generation###
def globaldataloader_generation(tokenizer, batch_size, len_max = 256): # length of sample is fixed to 256
    ###generate a simple training task###
    ###we fetch wikitext dataset from huggingface###
    from datasets import load_dataset
    rawDataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    def tok_fn(ex): 
        return tokenizer(ex["text"], return_attention_mask=False)
    def grp_fn(ex):
        concat = sum(ex["input_ids"], [])
        tot = len(concat) // len_max * len_max
        ids = [concat[i:i+len_max] for i in range(0, tot, len_max)]
        return {"input_ids": ids, "labels": [x[:] for x in ids]}
    
    ds = (rawDataset.map(tok_fn, batched=True, remove_columns=["text"])
              .map(grp_fn, batched=True))
    ds = ds.select(range(3*24)) #totally 72 samples
    ds.set_format("torch", columns=["input_ids", "labels"])
    
    return ds


# -------------------------------
### Helpers for unit test/grading(do not modify)
# -------------------------------
### these codes are for unit test. You don't have to understand
def flat_params(model: nn.Module):
    return torch.cat([p.detach().view(-1) for p in model.parameters()])

def sha256_chunks_u64(t: torch.Tensor):
    """Return 4x uint64 from SHA-256 of tensor bytes (small, fixed-size gather)."""
    # Hash raw bytes so dtypes such as bfloat16 also work with NumPy.
    arr = t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    h = hashlib.sha256(arr).digest()  # 32 bytes
    parts = [int.from_bytes(h[i:i+8], "little", signed=False) for i in range(0, 32, 8)]
    return torch.tensor(parts, dtype=torch.uint64)

def assert_models_identical(model: nn.Module, world: int, rank: int):
    vec = flat_params(model)
    sig_local = sha256_chunks_u64(vec).to(torch.int64)       
    gather_list = [torch.empty_like(sig_local) for _ in range(world)]
    dist.all_gather(gather_list, sig_local)                 #Noticed: the pytorch built-in all_gather is only used for unit test check(not allowed to use in your all-reduce implementaion)
    # Rank-0 checks all signatures equal
    if rank == 0:
        for i in range(1,dist.get_world_size()):
            ok = torch.equal(gather_list[0], gather_list[i])
            if not ok:
                raise AssertionError(f"Model checksums of rank0 differ across ranks:\n{i}")
        print("identical model test passed!:white_check_mark:")


###rest helper utils###
def shard_slice(n: int, rank: int, world: int):
    per = math.ceil(n / world)
    s, e = rank * per, min((rank + 1) * per, n)
    return s, e

###provided function: we do all-reduce sync for each layer###
###important: your ring_allreduce_ is called here###
###we do the all-reduce warpper for you### 
def allreduce_grads_ring_(model: nn.Module, world_size=None, rankid=None, opt=None):
    from q3 import ring_allreduce_
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads: return
    flat = _flatten_dense_tensors(grads)
    ring_allreduce_(flat, world_size = world_size, rankid = rankid)
    synced = _unflatten_dense_tensors(flat, grads)
    for g, s in zip(grads, synced):
        g.copy_(s)
    opt.step()

###main work###
def run_training(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.question == 2:
        from q2 import PS_grads_

    print("waiting distributed init setup...")
    ###set up global distributed config###
    dist.init_process_group("gloo", init_method="env://")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    print("dist group setup finished, begin model loading...")
    ###get pretrained qwen3-0.6b model from huggingface###
    device = torch.device("cpu")     
    name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    Qwenmodel = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)

    ###generate local dataset loader###
    len_max = 256
    ds = globaldataloader_generation(tokenizer, args.batch_size, len_max = len_max)

    start, end = shard_slice(len(ds), rank, world)
    local_ds = torch.utils.data.Subset(ds, range(start, end))
    loader = DataLoader(local_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)


    
    print(f"per device batch size is {args.batch_size}, global batch size is {args.batch_size*world}.")
    print(f"WikiText dataset ready: {len(ds)} sequences of length {len_max}.")
    print(f"each local dataset contains {len(local_ds)} samples in {len(loader)} batches.")
    print(f"node {rank} is assigned by sample's no from {start} to {end}.")

    ###get rid of randomness: do not modify this line###
    torch.manual_seed(42)
    
    Qwenmodel.to(device)
    Qwenmodel.train()

    opt = optim.Adam(Qwenmodel.parameters(), lr=1e-4)    #set optimizor
    ce_sum = nn.CrossEntropyLoss(reduction="sum")


    dist.barrier()

    time_stamp = []
    for epoch in range(args.epochs):
        run_loss_sum_local = 0.0
        run_tok_local = 0.0
        start_time = time.time()
        step = 0

        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            opt.zero_grad(set_to_none=True)

            out = Qwenmodel(input_ids=input_ids, use_cache=False, output_hidden_states=False)
            logits = out.logits  # [B, T, V]
            B, T, V = logits.shape
            # shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            local_token_amount = B * (T - 1)

            loss_sum = ce_sum(shift_logits.view(-1, V), shift_labels.view(-1))  # scalar SUM over tokens
            loss = loss_sum / local_token_amount
            loss.backward()

            if args.question == 2:
                ###do sync for each layers' parameter...###
                ###important: your code will be called here..
                pre_vec = flat_params(Qwenmodel).clone()
                print(f"epoch {epoch} step {step} begin gradient parameter server sync...")
                print("Using parameter server to sync...")
                PS_grads_(Qwenmodel, world_size = world, rankid =rank, opt=opt)
                print("parameter server sync finished.")
                # ---- ASSERT #2: Parameters actually changed (not a no-op) ----
                post_vec = flat_params(Qwenmodel)
                delta_norm = (post_vec - pre_vec).norm().item()
                if rank == 0:
                    # Non-finite or zero update usually means grads weren’t averaged/applied correctly
                    assert math.isfinite(delta_norm), "[PS check] non-finite parameter delta after PS step"
                    assert delta_norm > 0, "[PS check] zero parameter delta; PS step may not be applying updates"
            elif args.question == 3:
                ###do sync for each layers' parameter...###
                print(f"epoch {epoch} step {step} begin all reduce sync...")
                print("Using all reduce to sync...")
                allreduce_grads_ring_(Qwenmodel, world_size = world, rankid =rank, opt=opt)
                print("all reduce sync finished.")


            ### Logging (average loss per token over time)###
            run_loss_sum_local += loss_sum.item()
            run_tok_local += local_token_amount
        
        print(f"rank {rank} epoch {epoch} aggregated loss/token={run_loss_sum_local/run_tok_local:.4f}")
        dist.barrier()

        if rank == 0:
            total_time = time.time() - start_time
            time_stamp.append(total_time)
            print(f"\nEpoch {epoch+1} completed in {total_time:.2f}s")
            print(f"Average step speed: {step / total_time:.2f} steps/s")



        ###unit test###
        ###this test checks that if final models across nodes are identical###
    assert_models_identical(Qwenmodel, world = world, rank = rank)
    
    dist.destroy_process_group()


def close(actual, expected, exact=False):
    torch.testing.assert_close(actual, expected, rtol=0 if exact else 1e-4,
                               atol=0 if exact else 1e-5)


def unchanged(tensors, snapshots):
    for tensor, snapshot in zip(tensors, snapshots):
        close(tensor, snapshot, exact=True)


def check(label, action, rank, world):
    result, error = None, None
    try:
        result = action()
    except Exception as exc:
        error = f'rank {rank}: {type(exc).__name__}: {exc}'
    status = torch.tensor([error is None], dtype=torch.int32)
    statuses = [torch.empty_like(status) for _ in range(world)]
    dist.all_gather(statuses, status)  # Runner-only collective is allowed.
    passed = all(item.item() for item in statuses)
    if rank == 0:
        print(f"{'PASS' if passed else 'FAIL'} {label}", flush=True)
    if error:
        print(f'  {label}: {error}', flush=True)
    return passed, result


def bounds(width, rank, world):
    base, remainder = divmod(width, world)
    start = rank * base + min(rank, remainder)
    return start, start + base + int(rank < remainder)


def inputs(tokens, hidden, intermediate, seed):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(tokens, hidden, generator=generator),
            torch.randn(hidden, intermediate, generator=generator) / math.sqrt(hidden),
            torch.randn(intermediate, hidden, generator=generator) / math.sqrt(intermediate),
            torch.randn(tokens, hidden, generator=generator))


@contextmanager
def reference_sync(rank, world):
    """Isolate forward/backward from an unfinished student sum helper.

    Require use of that helper, and check that its argument is the correct
    local contribution. The all-reduce here is test infrastructure only.
    """
    original, calls = q4.sum_across_ranks, []
    def sync(tensor, passed_rank, passed_world):
        assert (passed_rank, passed_world) == (rank, world)
        calls.append(tensor.clone())
        result = tensor.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result
    q4.sum_across_ranks = sync
    try:
        yield calls
    finally:
        q4.sum_across_ranks = original


def autograd_reference(x, w1, w2, grad_output):
    # Autograd is used ONLY in the runner as an independent reference.
    with torch.enable_grad():
        xx, ww1, ww2 = [t.detach().clone().requires_grad_(True) for t in (x, w1, w2)]
        y = F.gelu(xx @ ww1) @ ww2
        gradients = torch.autograd.grad(y, (xx, ww1, ww2), grad_outputs=grad_output)
    return y.detach(), tuple(t.detach() for t in gradients)


def validate_cache(cache, x, w1, w2):
    assert isinstance(cache, tuple) and len(cache) == 5, 'cache must be (x,z,a,w1_local,w2_local)'
    z = x @ w1
    for actual, expected in zip(cache, (x, z, F.gelu(z), w1, w2)):
        close(actual, expected)


@torch.no_grad()
def component_checks(component, seed, rank, world, cases=None, results=None):
    # Equal shards, uneven shards (P>1), and the minimum one-feature shard.
    if cases is None:
        cases = [(1, 7, 3 * world), (5, 9, 2 * world + 1), (11, 6, world)]
    success = True
    for index, (tokens, hidden, intermediate) in enumerate(cases):
        x, w1, w2, grad_output = inputs(tokens, hidden, intermediate, seed + index)
        start, end = bounds(intermediate, rank, world)
        local1, local2 = w1[:, start:end].clone().contiguous(), w2[start:end].clone()
        yref, (dxref, dw1ref, dw2ref) = autograd_reference(x, w1, w2, grad_output)

        def shard_test():
            originals = (w1, w2)
            snapshots = tuple(t.clone() for t in originals)
            shards = q4.shard_weights(w1, w2, rank, world)
            unchanged(originals, snapshots)
            assert len(shards) == 2
            for actual, expected, full in zip(shards, (local1, local2), originals):
                close(actual, expected, exact=True)
                assert actual.is_contiguous(), 'Shard must be contiguous'
                assert actual.untyped_storage().data_ptr() != full.untyped_storage().data_ptr(), 'Shard must own its storage'
                assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size(), 'Shard must not retain full-weight storage'

        def sync_test():
            # Repeated calls check buffer lifetime and stale-message mistakes.
            for repeat in range(3):
                base = torch.arange(tokens * hidden, dtype=torch.float32).reshape(tokens, hidden)
                partial = base * (rank + 1) + repeat
                before = partial.clone()
                result = q4.sum_across_ranks(partial, rank, world)
                close(result, base * (world * (world + 1) / 2) + repeat * world)
                close(partial, before, exact=True)
                assert result.untyped_storage().data_ptr() != partial.untyped_storage().data_ptr(), 'Sum result must own its storage'

        def forward_test():
            arguments = (x.clone(), local1.clone(), local2.clone())
            snapshots = tuple(t.clone() for t in arguments)
            with reference_sync(rank, world) as calls:
                output, cache = q4.mlp_forward(*arguments, rank, world)
            assert len(calls) == 1, 'Forward must use sum_across_ranks exactly once'
            close(calls[0], F.gelu(x @ local1) @ local2)
            close(output, yref)
            validate_cache(cache, *arguments)
            unchanged(arguments, snapshots)

        def backward_test():
            # Canonical cache makes this independent of the forward implementation.
            z = x @ local1
            cache = (x.clone(), z, F.gelu(z), local1.clone(), local2.clone())
            upstream = grad_output.clone()
            originals = (upstream, *cache)
            snapshots = tuple(t.clone() for t in originals)
            with reference_sync(rank, world) as calls:
                dx, dw1, dw2 = q4.mlp_backward(upstream, cache, rank, world)
            assert len(calls) == 1, 'Backward must sum input-gradient contributions exactly once'
            _, (local_dx, _, _) = autograd_reference(x, local1, local2, grad_output)
            close(calls[0], local_dx)
            close(dx, dxref)
            close(dw1, dw1ref[:, start:end])
            close(dw2, dw2ref[start:end])
            unchanged(originals, snapshots)

        for name, action in [('shard', shard_test), ('sync', sync_test),
                             ('forward', forward_test), ('backward', backward_test)]:
            if component in ('all', name):
                passed, _ = check(f'case {index + 1}: {name}', action, rank, world)
                success = passed and success
                if results is not None:
                    results[name] = results.get(name, True) and passed
    return success


@torch.no_grad()
def combined_checks(seed, rank, world, shape=None):
    """Check the actual shard/sum/forward/backward functions together."""
    x, w1, w2, upstream = inputs(*(shape or (5, 7, 3 * world + 1)), seed + 100)
    start, end = bounds(w1.shape[1], rank, world)
    yref, (dxref, dw1ref, dw2ref) = autograd_reference(x, w1, w2, upstream)
    passed, shards = check('combined: initialize shards',
                           lambda: q4.shard_weights(w1.clone(), w2.clone(), rank, world), rank, world)
    if not passed:
        return False
    def forward():
        output, cache = q4.mlp_forward(x, *shards, rank, world)
        close(output, yref)
        validate_cache(cache, x, *shards)
        return cache
    passed, cache = check('combined: forward', forward, rank, world)
    if not passed:
        return False
    def backward():
        dx, dw1, dw2 = q4.mlp_backward(upstream, cache, rank, world)
        close(dx, dxref)
        close(dw1, dw1ref[:, start:end])
        close(dw2, dw2ref[start:end])
    passed, _ = check('combined: backward', backward, rank, world)
    return passed


def run_q4_diagnostics(args):
    global q4
    q4 = importlib.import_module('q4')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=30))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        success = component_checks(args.component, args.seed, rank, world)
        if args.component == 'all' and success:
            success = combined_checks(args.seed, rank, world)
        elif args.component == 'all' and rank == 0:
            print('SKIP combined forward/backward: finish the component checks first.', flush=True)
        if rank == 0:
            print(f"Q4 {'PASSED' if success else 'FAILED'} ({world} processes, {args.component}, seed={args.seed})", flush=True)
    finally:
        dist.destroy_process_group()
    if not success:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', type=int, choices=[2, 3, 4], default=3,
                        help='2: parameter-server training; 3: ring-all-reduce training; 4: tensor-parallel checks (default: 3)')
    parser.add_argument('--batch_size', type=int, default=int(os.getenv('BATCH_SIZE', 8)),
                        help='Q2/Q3 batch size per rank (default: 8)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Q2/Q3 training epochs (default: 1)')
    parser.add_argument('--component', choices=['all', 'shard', 'sync', 'forward', 'backward'], default='all',
                        help='Q4 component to check (default: all)')
    parser.add_argument('--seed', type=int, default=498, help='Q4 test seed (default: 498)')
    args = parser.parse_args()
    if args.question in (2, 3):
        if args.batch_size <= 0:
            parser.error('--batch_size must be positive')
        if args.epochs < 0:
            parser.error('--epochs must be nonnegative')
    if args.question == 4:
        run_q4_diagnostics(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
