# Homework 1 Coding Instructions

This README covers all three coding questions. Submit `q2.py`, `q3.py`, and
`q4.py` with your short-answer responses as described in the assignment.

- [Q2: Parameter Server](#q2-parameter-server)
- [Q3: Ring All-Reduce](#q3-ring-all-reduce)
- [Q4: Tensor-Parallel MLP](#q4-tensor-parallel-mlp)

All three questions use `run.py` and `glooHelper.sh`. Select the question with `-q 2`, `-q 3`, or `-q 4` when using `glooHelper.sh`.

## Coding grading: 30% public / 70% hidden

Each coding question and each of its components uses **30% public-test credit
and 70% hidden-test credit**.

| Question | Total | Public tests (30%) | Hidden tests (70%) |
| --- | ---: | ---: | ---: |
| Coding Q2 | 10 | 3 | 7 |
| Coding Q3 | 30 | 9 | 21 |
| Coding Q4 | 25 | 7.5 | 17.5 |


There are three public cases and seven hidden cases per question. Each case
contributes 10% of each component's points; fractional points are retained.
You can earn component credit without passing every part of a case.
Hidden cases follow the same documented contracts and vary inputs, dimensions,
and supported process counts. They do not require additional implementation
features.

From this homework directory, score the public tests with CPU PyTorch:

```bash
python grade_public.py --question 2 --submission .
python grade_public.py --question 3 --submission .
python grade_public.py --question 4 --submission .
```

Use `--question all` to check all three, `--output results.json` for a separate
report, and optionally `--jobs 2` to run cases concurrently. Each case launches
its own local process group and has a 60-second timeout. The JSON report
contains per-component points; the adjacent `.logs` directory contains test
output. A successful grader invocation means the report was produced—read
the scores to see which checks passed.

The scored Q2/Q3 tests use small synthetic models/tensors and do not download
Qwen or WikiText. In addition to the public and hidden grading tests, course
staff will run your submitted `q2.py`, `q3.py`, and `q4.py` with our official
copy of `run.py` to verify correctness in the provided workflow: Q2/Q3 must
complete Qwen training and pass the runner's checks, and Q4 must pass its
component and combined forward/backward checks. If your implementation fails the official `run.py` checks, **50% of the
affected coding question's total points will be deducted**: **5 points for
Q2, 15 points for Q3, or 12.5 points for Q4**. This deduction is applied once
per affected question, after combining its public and hidden test scores,
with a minimum final score of zero. Multiple failed checks within the same
question do not incur additional runner-failure deductions.

Use `run.py --question 2`, `run.py --question 3`, or `run.py --question 4`
through the launch commands below to check your implementation before submitting.
These additional checks do not award points beyond the stated public/hidden
allocation. Official grading uses staff-controlled runners and test files
with your submitted implementations. Do not submit or modify grading files.

## Setup and running Q2–Q4

All three questions use CPU PyTorch and the Gloo distributed backend, with
one process on each of three machines. The setup and launcher instructions
below apply to Q2, Q3, and Q4.

Q2/Q3 additionally need Transformers and Datasets to load Qwen3-0.6B and
WikiText. Q4 and the public grading tests use only small local tensors and
need no model or dataset downloads.

> **Start early!** Q2/Q3 training runs take several minutes. When multiple
> students share a cluster, queueing will slow things down.

### Experiment Setup Instructions - Cloudlab

We created multiple clusters and assigned each student to **one** cluster. We have assigned students to clusters based on their last name. **Please run your experiments only on your assigned cluster** to avoid overloading others. If students run into any issues due to the clusters getting overloaded, please email us, and we will assign you to new cluster that has less congestion.

Cluster Assignments:

- HW1-Cluster-1: Aenlle–Guo
- HW1-Cluster-2: Gupta–Leong
- HW1-Cluster-3: Li–Park
- HW1-Cluster-4: Patel–Tapia
- HW1-Cluster-5: Tripathi–Zhou

1. **Familiarize yourself with CloudLab**

   * Go to **`Experiment → Topology View`** to see the network layout (triangle topology).
   * Click a node, then **`Shell`** to you can open an in-browser SSH session.
   * To SSH from your own terminal, upload your key via the top-right menu **Manage SSH Keys**. Then use **`Experiment → List View`** to copy the SSH command for a node.

2. **Python environment**

   * We recommend using **conda** with **Python ≥ 3.10.12** to avoid dependency issues.
   * Install CPU PyTorch for all questions. Transformers and Datasets are
     additionally required for Q2/Q3 training:

     ```bash
     sudo apt-get update
     sudo apt-get install -y python3-pip
     pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
     pip install transformers datasets
     echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
     source ~/.bashrc
     ```

3. **Know PyTorch Gloo**

   * The project uses PyTorch’s **`gloo`** distributed backend.
   * To ensure nodes communicate over the experiment LAN, set the network interface for Gloo on **each** node:

      1. Run `ifconfig` and identify the interface connected to the experiment network (typically named like `en*`, e.g., `enp4s0f1`, with a `10.x.x.x` address. There may be several candidates, you can choose one).
      2. For each node, export the interface name and master address, which you can choose, but for simplicity, we recommend choosing node-0's address.
        ```bash
        export GLOO_SOCKET_IFNAME=enp4s0f1
        export MASTER_ADDR=10.x.x.x
        ```
      * **Important:** If this variable is not set correctly, the distributed job maybe hang forever, and you will need to close the terminal session and reopen. 
      * **Note** Use the same `MASTER_ADDR` on all nodes. Select `GLOO_SOCKET_IFNAME` separately on each node using its actual experiment interface; interface names may be the same or different.
      * Example: Let's use node-0 as the master-node. In this example, `ifconfig` on node-0 gives us a node interface `enp94s0f0` with an inet of `10.10.2.2`, and node interface `enp94s0f1` with an inet of `10.10.1.2`. `ifconfig` on node-1 gives us a node interface `enp94s0f0` with an inet of `10.10.3.2`, and node interface `enp94s0f1` with an inet of `10.10.1.1`. `ifconfig` on node-2 gives us a node interface `enp94s0f0` with an inet of `10.10.3.1`, and node interface `enp94s0f1` with an inet of `10.10.2.1`. We set the master address in all 3 nodes with `export MASTER_ADDR=10.10.2.2`. We set the node-0's gloo socket with `export GLOO_SOCKET_IFNAME=enp94s0f0`, and node-1 and node-2's gloo sockets with `export GLOO_SOCKET_IFNAME=enp94s0f1`.
     3. To check if this was done properly, you can run `ping -I $GLOO_SOCKET_IFNAME $MASTER_ADDR` on all nodes, and it should run without issue.
### Run on three machines

Put the same homework files on all three machines and run commands from the
homework directory. Place your NetID in `netid.txt`, with identical contents
on every machine. Use the same `MASTER_ADDR` and each machine's experiment
interface as described above.

The helper derives a shared port from `netid.txt`. To choose a different port,
set the same `MASTER_PORT` on every machine. If a port is busy, choose another
shared port; the helper never changes ports independently.

Set `QUESTION` to `2`, `3`, or `4` on each machine, then launch the command for
that machine's rank. All three commands must be running:

```bash
QUESTION=2  # Change to 3 or 4 on every machine for that question.

# Machine 0
bash glooHelper.sh -n 3 -P 1 -r 0 -q "$QUESTION" -s ./run.py

# Machine 1
bash glooHelper.sh -n 3 -P 1 -r 1 -q "$QUESTION" -s ./run.py

# Machine 2
bash glooHelper.sh -n 3 -P 1 -r 2 -q "$QUESTION" -s ./run.py
```

- `-n`: number of machines; use `3`.
- `-P`: processes per machine; use `1`.
- `-r`: this machine's node rank (`0`, `1`, or `2`); rank 0 hosts rendezvous.
- `-q`: question number (`2`, `3`, or `4`).
- `-s`: shared runner, `./run.py`.
- `-e`: training epochs for Q2/Q3, default `1`; not used for Q4.

Use identical options on every machine except `-r`. Additional runner options
go after `--`; Q4's component and seed options are described in its section.

For Q2/Q3, you can check distributed initialization and model/dataset loading
before implementing synchronization by adding `-e 0`. This skips training and
does not establish that your synchronization code is correct.

### Optional local diagnostics

For debugging without a cluster, launch three processes on one machine:

```bash
torchrun --standalone --nproc-per-node=3 run.py --question 2
torchrun --standalone --nproc-per-node=3 run.py --question 3
torchrun --standalone --nproc-per-node=3 run.py --question 4
```

Run the command for the question you are working on. Q2/Q3 still load and train
Qwen, so allow enough CPU memory for three model replicas. Q4 uses small tensors;
also check its supported one- and four-process cases by changing
`--nproc-per-node` to `1` or `4`. Set `GLOO_SOCKET_IFNAME` to a valid local
interface, or unset a stale value copied from another machine.

`grade_public.py` launches its scoring cases locally. Diagnostic runs do not
award additional points.

### Q2/Q3 training workflow and tools

For Q2 and Q3, your coding work is in `q2.py` and `q3.py`. However, understanding the overall training flow will help. In `run.py`, we prepare a WikiText dataset for Qwen3 fine-tuning training. Each sample has a fixed length of **256**. The total dataset size is **3 × 24** (so each node gets **24** samples). We fix the batch size to **8**, so each epoch has **24 / 8 = 3** steps. Training defaults to one epoch. The `-e` flag selects the number of epochs; `-q 2` uses PS throughout the run and `-q 3` uses All-Reduce throughout the run. After each epoch, the aggregated loss per token is printed for debugging.

1. **Read the comments.** We’ve added essential comments to guide you through the training flow. Also review the slides on **All-Reduce** and **Parameter Server**. In `run.py`, focus on the provided `allreduce_grads_ring_` wrapper and the training loop in `run_training`—that’s where the sync method is applied.

2. **Know `nn.Module` basics.** You should be comfortable extracting parameters and grads:

   ```python
   params = [p for p in model.parameters() if p.grad is not None]
   grads  = [p.grad for p in params]
   ```

   See the PyTorch documentation for details on modules, parameters... https://docs.pytorch.org/docs/stable/pytorch-api.html

3. **Available tools.** You **may not** use PyTorch’s direct collectives (e.g., `dist.all_reduce`) for this assignment. With `import torch.distributed as dist` available, implement your own sync using **non-blocking P2P**:

   ```python
   # Rank 0:
   send_buf = torch.tensor([1, 2, 3])
   s = dist.isend(send_buf, dst=2)   # send to rank 2
   s.wait()

   # Rank 2:
   recv_buf = torch.empty(3, dtype=torch.long)
   r = dist.irecv(recv_buf, src=0)   # receive from rank 0
   r.wait()

   # Always wait before reusing/read/writing the buffers
   # after wait, rank 2's recv_buf == tensor([1, 2, 3])
   ```

   Also, for easier chunking/merging, use:

   * `torch._utils._flatten_dense_tensors(list_of_tensors)`
   * `torch._utils._unflatten_dense_tensors(flat_tensor, like_list)`
     These help turn a list of high-dim tensors into a single 1-D buffer (and back).

Matching final parameters is a basic integration check; use the public grader
to check numerical correctness and component scores. You may modify `run.py`
for your own Q2/Q3 debugging, but submit only the question implementation files.

## Q2: Parameter Server

Implement `server` and `worker` in `q2.py` for three-node data-parallel training.
The provided `PS_grads_` wrapper selects parameters with gradients and invokes
the appropriate function for each rank.

- **Server (rank 0):** receive worker gradients, include its own gradients,
  average across all ranks, take one optimizer step, and send the updated
  parameters to every worker.
- **Workers:** send gradients to rank 0, receive its updated parameters, and
  copy them into the local model. Workers do not take an optimizer step.

Use the shared launch instructions with `-q 2`. `run.py` uses the parameter
server throughout training, checks that updates are finite and nonzero, and
checks that final model parameters match across ranks. Submit `q2.py`.

## Q3: Ring All-Reduce

Implement `ring_allreduce_`, `reduce_scatter`, and `all_gather` in `q3.py` for
three-node data-parallel training. Pad tensors when needed to form equal-sized
chunks, perform the ring reduce-scatter and all-gather phases, remove padding,
and write the averaged result back into the original tensor.

You may adjust the helper signatures of `reduce_scatter` and `all_gather` to
support your implementation. Preserve the `ring_allreduce_` interface.
The provided `allreduce_grads_ring_` wrapper in `run.py` packs gradients,
calls your implementation, restores gradients, and takes an optimizer step
on every rank.

Use the shared launch instructions with `-q 3`. `run.py` uses ring all-reduce
throughout training and checks that final model parameters match across ranks.
Submit `q3.py`.

## Q4: Tensor-Parallel MLP

Implement `shard_weights`, `sum_across_ranks`, `mlp_forward`, and `mlp_backward`
in `q4.py`. Partition the MLP weights across ranks, compute local forward and
backward operations, and sum partial outputs and input gradients. Weight
gradients stay local. This question is independent of your Q2/Q3 solutions
and uses small CPU tensors; no Qwen model, dataset, or optimizer update is needed.

Preserve the four function signatures and the provided GeLU derivative.
Follow the assignment's partitioning rules, function contracts, and supplied
backward formulas. Only `sum_across_ranks` may communicate, using rank-0
aggregation and distribution with `dist.isend`, `dist.irecv`, and `wait()`.
Do not use built-in collectives, autograd, or reconstruct full weights during
forward/backward computation.

Use the shared launch instructions with `-q 4`. `run.py` checks each component
against a reference and then checks your functions together. To check one
component, append `-- --component shard`, `sync`, `forward`, or `backward`
to the launch command on every machine. Submit `q4.py`.
