###Q3: allreduce###
###please implement ring_allreduce method, using  pytorch's dist method is not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def reduce_scatter(chunks, tmp, world, rank, left, right):
    # your code here: follow slides instruction: do counter-clockwise iteration
    buf = torch.zeros_like(chunks[0])

    for i in range(world - 1):
        send_idx = (rank - i + world) % world
        recv_idx = (rank - i - 1 + world) % world

        send_req = dist.isend(tensor=chunks[send_idx], dst=right)
        recv_req = dist.irecv(tensor=buf, src=left)
        recv_req.wait()

        chunks[recv_idx] += buf
        send_req.wait()
    # --- end of code

    return
        
def all_gather(chunks, tmp, current, world, rank, left, right):
    # --- your code here: follow slides instruction: do counter-clockwise iteration ---
    buf = torch.zeros_like(chunks[0])

    for i in range(world - 1):
        send_idx = (rank - i - 1 + world) % world
        recv_idx = (rank - i - 2 + world) % world

        send_req = dist.isend(tensor=chunks[send_idx], dst=right)
        recv_req = dist.irecv(tensor=buf, src=left)
        recv_req.wait()

        chunks[recv_idx].copy_(buf)
        send_req.wait()
    # --- end of code ---
    return

def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce (SUM, optional average) using isend/irecv."""
    world = world_size
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world

    # --- your code here: we cannot divide flat into 3 pieces evenly ---
    pad = chunk * world - n
    padded_flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]
    # --- end of code ---

    # your code here: call reduce_scatter and all_gather
    tmp = torch.empty_like(chunks[0])
    reduce_scatter(chunks, tmp, world, rank, left, right)
    current = (rank - (world - 1)) % world
    all_gather(chunks, tmp, current, world, rank, left, right)
    # --- end of code ---
    
    # stitch & unpad  
    padded_flat /= world
    tensor.view(-1).copy_(padded_flat[:n])
    return