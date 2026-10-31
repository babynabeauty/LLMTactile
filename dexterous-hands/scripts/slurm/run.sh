#!/bin/bash
# Generic SLURM wrapper. `scripts/expand_distill_config.py --submit` launches one of these
# per sweep cell; it also works standalone for a single run:
#
#   sbatch scripts/slurm/run.sh train --task=in_hand_repose --robot=wuji ...
#
# Adjust the SBATCH preamble for your cluster.

#SBATCH --output=logs/slurm/%j-%x.out
#SBATCH --job-name=tactile_genesis
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-0:00:00
#SBATCH --mem=32G

set -e

if [ $# -lt 1 ]; then
    echo "Usage: sbatch scripts/slurm/run.sh <train|play|path/to/script.py> <args...>" >&2
    exit 1
fi

echo "Running on node: $(hostname)"
echo "Args: $*"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

MODE=$1
shift

case $MODE in
    train) python main.py --mode=train "$@" ;;
    play)  python main.py --mode=play "$@" ;;
    *)     python "$MODE" "$@" ;;
esac
