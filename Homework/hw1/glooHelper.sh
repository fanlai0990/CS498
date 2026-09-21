#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 -n <nodes> -P <processes_per_node> -r <node_rank> -q <2|3|4> -s <script.py> [-e <epochs>] [-- <script args...>]

Options:
  -n    Number of machines (use 3 for the CloudLab workflow)
  -P    Processes per machine (use 1 for the CloudLab workflow)
  -r    This machine's rank, from 0 to nodes-1
  -q    Question: 2 (parameter server), 3 (ring all-reduce), 4 (tensor-parallel MLP)
  -s    Runner path (normally ./run.py)
  -e    Q2/Q3 training epochs, default 1; not used for Q4 checks
  -h    Show this help

Environment on every machine:
  MASTER_ADDR          Reachable address of the rank-0 machine (required)
  GLOO_SOCKET_IFNAME    This machine's experiment-network interface
  MASTER_PORT          Optional shared port; otherwise derived from netid.txt

Use identical netid.txt contents and the same MASTER_ADDR/MASTER_PORT on all
machines. Ports are never changed automatically on individual machines.

Examples (run with -r 0, 1, and 2 on the corresponding machines):
  bash $0 -n 3 -P 1 -r 0 -q 2 -e 1 -s ./run.py
  bash $0 -n 3 -P 1 -r 0 -q 4 -s ./run.py
  bash $0 -n 3 -P 1 -r 0 -q 4 -s ./run.py -- --component backward
EOF
}

die() { echo "ERROR: $*" >&2; exit 2; }
NNODES='' NPROC='' NODE_RANK='' SCRIPT='' EPOCHS=1 QUESTION=2
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -n|-P|-r|-e|-q|-s)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      case "$1" in
        -n) NNODES="$2" ;;
        -P) NPROC="$2" ;;
        -r) NODE_RANK="$2" ;;
        -e) EPOCHS="$2" ;;
        -q) QUESTION="$2" ;;
        -s) SCRIPT="$2" ;;
      esac
      shift 2 ;;
    *) die "Unknown option: $1 (use --help)" ;;
  esac
done
SCRIPT_ARGS=("$@")
[[ "$NNODES" =~ ^[1-9][0-9]*$ ]] || die '-n must be a positive integer'
[[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || die '-P must be a positive integer'
[[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || die '-r must be a nonnegative integer'
(( NODE_RANK < NNODES )) || die '-r must be less than -n'
[[ "$EPOCHS" =~ ^(0|[1-9][0-9]*)$ ]] || die '-e must be a nonnegative integer'
[[ "$QUESTION" =~ ^[234]$ ]] || die '-q must be 2, 3, or 4'
[[ -n "${MASTER_ADDR:-}" ]] || die 'Set MASTER_ADDR to the rank-0 machine address on every machine'
[[ -f "$SCRIPT" ]] || die "Runner not found: $SCRIPT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${MASTER_PORT:-}" ]]; then
  HASH_SRC="${SCRIPT_DIR}/netid.txt"
  [[ -f "$HASH_SRC" ]] || die "Create $HASH_SRC with the same NetID on every machine"
  if command -v sha256sum >/dev/null 2>&1; then
    HASH_HEX="$(sha256sum "$HASH_SRC" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    HASH_HEX="$(shasum -a 256 "$HASH_SRC" | awk '{print $1}')"
  else
    die 'sha256sum or shasum is required'
  fi
  MASTER_PORT=$(( (0x${HASH_HEX:0:8} % 40000) + 20000 ))
  PORT_SOURCE='netid.txt'
else
  PORT_SOURCE='MASTER_PORT override'
fi
[[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] || die 'MASTER_PORT must be an integer from 1 to 65535'
(( MASTER_PORT <= 65535 )) || die 'MASTER_PORT must be an integer from 1 to 65535'
# Every launcher must choose the SAME port. Local port probing can otherwise
# cause the nonzero ranks to disagree with an already-running rank-0 launcher.
export MASTER_PORT

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT ($PORT_SOURCE)"
echo "NNODES=$NNODES PROCESSES_PER_NODE=$NPROC NODE_RANK=$NODE_RANK QUESTION=$QUESTION"
if [[ "$QUESTION" == 4 ]]; then
  echo 'Running Q4 CPU checks (no model/dataset downloads; epochs do not apply).'
  RUN_ARGS=(--question "$QUESTION")
else
  RUN_ARGS=(--question "$QUESTION" --epochs "$EPOCHS")
fi

exec torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node="$NPROC" \
  --node_rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  "$SCRIPT" "${RUN_ARGS[@]}" "${SCRIPT_ARGS[@]}"
