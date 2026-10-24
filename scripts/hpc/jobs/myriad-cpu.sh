#!/bin/bash -l
# Job script of Myriad CPU (SGE): CPU-only D nodes, 36 cores (Xeon Gold 6240, AVX-512) and
# 192 GB each, no GPU (ADR-0014). No -l gpu request; the job sets GRIDCL_TEST_DEVICE=cpu.
# The CPU runtime's threads follow the job's slots, so CPU work asks for --cores 36.
# remote-test.sh renders it for every submission (lib/submit.sh): it replaces the @...@
# placeholders, replaces the line #@REQUESTS@ with the cores request (CORES_REQUEST, for
# --cores or --ranks) and the memory request (MEM_REQUEST) of targets.env, and appends the
# job settings and jobs/common.sh. The #$ lines stay in the submitted script, before its
# first command, so `scriptfor <id>` shows them. Keep every line up to #@REQUESTS@ a comment.
#$ -S /bin/bash
#$ -N @NAME@
#$ -ac allow=D
#$ -l h_rt=@TIME@
#$ -wd @TREE@
#$ -o @TREE@/results/sge-$JOB_ID.log
#$ -j y
#@REQUESTS@
