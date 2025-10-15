#!/bin/bash

# --- Configuration ---
set -e

USER="alanluo3"
SSH_KEY="~/.ssh/id_ed25519"
PROJECT_DIR="$HOME/CS498/Homework/hw1"
DEST_PATH="~"

HOSTS=(
    "c220g5-111203.wisc.cloudlab.us" # Node 1
    "c220g5-111215.wisc.cloudlab.us" # Node 2
)

echo "Copying from $PROJECT_DIR to ${#HOSTS[@]} nodes..."

for i in "${!HOSTS[@]}"; do
    (
        HOST="${HOSTS[$i]}"
        echo "=================================================="
        echo "Starting VM $(($i + 1)): $HOST"
        
        # copy PROJECT_DIR from local machine to the remote node.
        echo "- Copying local '$PROJECT_DIR' directory to $HOST:$DEST_PATH..."
        scp -q -r -i "$SSH_KEY" "$PROJECT_DIR" "$USER@$HOST:$DEST_PATH"
        
        echo "Finished VM $(($i + 1)): $HOST"
        echo "=================================================="
    ) &
done

echo "Waiting for all VMs to finish..."
wait
echo "All VMs configured."
