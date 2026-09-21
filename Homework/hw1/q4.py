###Q4: Tensor-Parallel MLP###
###please implement weight partitioning, forward, and backward computation###
###use isend/irecv only in sum_across_ranks; built-in collectives are not allowed###
###do not use autograd or optimizers; see the assignment for contracts and README.md for running instructions###
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F


def gelu_derivative(z):
    ###provided helper: derivative of exact GeLU###
    ###do not modify this function###
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))) + z * torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)


@torch.no_grad()
def shard_weights(w1, w2, rank, world_size):
    # ---- partition the intermediate features across ranks ----
    # w1 has shape [H, F]; w2 has shape [F, H]; F >= world_size.
    # If F is not divisible by world_size, assign one extra feature to
    # each of the first F % world_size ranks. Use contiguous rank order.
    # Return independent, contiguous w1/w2 shards; do not modify full weights.
    #                                                                   #
    # your code here: find this rank's feature range, including remainder #
    #                columns, and copy the matching w1/w2 shards          #
    #                                                                   #
    raise NotImplementedError("Implement shard_weights")


@torch.no_grad()
def sum_across_ranks(tensor, rank, world_size):
    # ---- sum contributions and distribute the result ----
    # All ranks provide tensors with the same shape and dtype.
    # Return a fresh tensor containing the full SUM on every rank.
    # Do not change the input or divide by world_size.
    # With one rank, return an independent copy without communication.
    # Only this function may communicate: use isend/irecv and wait().
    #                                                                   #
    # your code here: handle the single-rank case                        #
    #                                                                   #

    # ---- rank 0: aggregate contributions, then send the sum ----
    #                                                                   #
    # your code here: include rank 0's input and receive from workers;    #
    #                send the completed sum to every worker              #
    #                                                                   #

    # ---- other ranks: send the local contribution, receive the sum ----
    #                                                                   #
    # your code here: send to rank 0 and receive the completed sum        #
    #                Wait before reading/reusing buffers or returning.   #
    #                                                                   #
    raise NotImplementedError("Implement sum_across_ranks")


@torch.no_grad()
def mlp_forward(x, w1_local, w2_local, rank, world_size):
    # ---- compute this rank's part of the MLP ----
    # x: [T, H]; w1_local: [H, F_r]; w2_local: [F_r, H].
    # Use the default F.gelu (exact mode); do not modify the inputs.
    # Call sum_across_ranks once to obtain the full output [T, H].
    # Return (output, cache), with cache in this exact order:
    # (x, z, a, w1_local, w2_local). Cached references are allowed.
    # ---- local up-projection, GeLU, and down-projection ----
    #                                                                   #
    # your code here: compute z, a, and the local partial output          #
    #                                                                   #

    # ---- combine outputs and save values for backward ----
    #                                                                   #
    # your code here: call sum_across_ranks and return output with cache  #
    #                                                                   #
    raise NotImplementedError("Implement mlp_forward")


@torch.no_grad()
def mlp_backward(grad_output, cache, rank, world_size):
    # ---- compute local gradients and synchronize the input gradient ----
    # grad_output is the same full dL/dY on every rank; do not scale by P.
    # Use gelu_derivative and the backward formulas in the assignment.
    # Call sum_across_ranks once for grad_x; weight gradients stay local.
    # Do not modify grad_output or the cache, including its saved weights.
    # Return (grad_x, grad_w1_local, grad_w2_local).
    # ---- unpack the cache and compute the local weight gradients ----
    #                                                                   #
    # your code here: compute grad_w2, grad_z, and grad_w1 manually        #
    #                Use the supplied gelu_derivative helper.            #
    #                                                                   #

    # ---- combine input-gradient contributions ----
    #                                                                   #
    # your code here: compute local grad_x, sum it across ranks, and      #
    #                return grad_x with the two local weight gradients   #
    #                                                                   #
    raise NotImplementedError("Implement mlp_backward")
