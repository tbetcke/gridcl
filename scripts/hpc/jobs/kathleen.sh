#!/bin/bash -l
# Job script of Kathleen (Slurm): whole diskless nodes of 2 × 20 cores (Xeon Gold 6248,
# Cascade Lake, AVX-512), 192 GB and Omni-Path, no GPU; at most two nodes per job (ADR-0016).
# The job sets GRIDCL_TEST_DEVICE=cpu. remote-test.sh renders it for every submission
# (lib/submit.sh): it replaces the @...@ placeholders (@QOS@: singlenode for one node, test for
# two nodes up to 1 h, small beyond), and appends the job settings and jobs/common.sh. The
# #SBATCH lines must come before the first command; the commands below them set the parts of
# the environment that hold on every Kathleen node (the module lines, the Rust toolchain and
# the gcc runtime on LD_LIBRARY_PATH come from KATHLEEN_MODULES in targets.env).
# 40 tasks per node, so that mpirun inside the job has a slot per core.
#SBATCH --job-name=@NAME@
#SBATCH --partition=kathleen
#SBATCH --qos=@QOS@
#SBATCH --nodes=@NODES@
#SBATCH --ntasks-per-node=40
#SBATCH --time=@TIME@
#SBATCH --chdir=@TREE@
#SBATCH --output=@TREE@/results/slurm-%j.log
#@REQUESTS@
# shellcheck disable=SC2034 # read by the appended jobs/common.sh
JOB_ID=$SLURM_JOB_ID
# In the batch step Slurm sets SLURM_STEP_ID=-5; Open MPI 4.1 then takes every MPI singleton
# (a test executable that cargo test starts) for an srun launch, and it aborts in MPI_Init.
case ${SLURM_STEP_ID:-} in
    -5 | 4294967291) unset SLURM_STEP_ID SLURM_STEPID ;;
esac
# Open MPI's session directories in the node's RAM, never on Lustre; PSM2's expected-receive
# (TID) path delivered wrong data between nodes (measured by the fmm project; ADR-0016).
export OMPI_MCA_orte_tmpdir_base=/dev/shm PSM2_TID=0
# Slurm gives the batch step every hardware thread (80 per node); the CPU runtime takes one
# thread per CPU it may use, so the batch shell, and everything it starts, is bound to the
# first hardware thread of each physical core (40 per node). mpirun binds its ranks itself.
gridcl_kathleen_cores=$(lscpu -p=CPU,CORE 2>/dev/null | grep -v '^#' | sort -t, -k2,2n -u |
    cut -d, -f1 | sort -n | paste -sd, -)
if [ -n "$gridcl_kathleen_cores" ]; then
    taskset -cp "$gridcl_kathleen_cores" $$ >/dev/null 2>&1 || true
fi
