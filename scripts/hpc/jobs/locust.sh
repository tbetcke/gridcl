#!/bin/bash
# Job script of Locust (direct): one NVIDIA GH200, Grace aarch64 CPU and Hopper GPU (sm_90).
# Locust has no scheduler and is shared with other users. remote-test.sh renders this file for
# every run (lib/submit.sh), appends the job settings and jobs/common.sh, and starts it on the
# node with timeout through ssh (lib/direct.sh); JOB_ID comes from the runner. The job records
# node-load snapshots before and after the command (ADR-0002). Module lines and CUDA_PATH come
# from targets.env (LOCUST_MODULES, LOCUST_CUDA_PATH); nothing is reserved, so --cores and the
# SGE requests do not apply.
