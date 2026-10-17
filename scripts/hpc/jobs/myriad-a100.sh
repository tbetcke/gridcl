#!/bin/bash -l
# Job script of Myriad A100 (SGE): L nodes, 36 cores and 4 x A100 40 GB (sm_80) each.
# remote-test.sh renders it for every submission (lib/submit.sh): it replaces the @...@
# placeholders, replaces the line #@REQUESTS@ with the cores request (CORES_REQUEST, for
# --cores or --ranks) and the memory request (MEM_REQUEST) of targets.env, and appends the
# job settings and jobs/common.sh. The #$ lines stay in the submitted script, before its
# first command, so `scriptfor <id>` shows them. Keep every line up to #@REQUESTS@ a comment.
#$ -S /bin/bash
#$ -N @NAME@
#$ -l gpu=@GPUS@
#$ -ac allow=L
#$ -l h_rt=@TIME@
#$ -wd @TREE@
#$ -o @TREE@/results/sge-$JOB_ID.log
#$ -j y
#@REQUESTS@
