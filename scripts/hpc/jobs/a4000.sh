#!/bin/bash
# Job script of a4000 (direct): a workstation with one NVIDIA RTX A4000 (sm_86), x86-64 (ADR-0037).
# The workstation has no scheduler and is shared with other users. remote-test.sh renders this
# file for every run (lib/submit.sh), appends the job settings and jobs/common.sh, and starts it
# on the node with timeout through ssh (lib/direct.sh); JOB_ID comes from the runner. The job records
# node-load snapshots before and after the command (ADR-0002). Module lines and CUDA_PATH come
# from targets.env (A4000_MODULES, A4000_CUDA_PATH); nothing is reserved, so --cores and the
# SGE requests do not apply.
