#!/bin/bash -l
# Job script of Myriad V100 (SGE): E and F nodes, 36 cores and 2 x V100 (sm_70) each.
# remote-test.sh renders it for every submission (lib/submit.sh): it replaces the @...@
# placeholders, replaces the line #@REQUESTS@ with the cores request (CORES_REQUEST, for
# --cores or --ranks) and the memory request (MEM_REQUEST) of targets.env, and appends the
# job settings and jobs/common.sh. The #$ lines stay in the submitted script, before its
# first command, so `scriptfor <id>` shows them. Keep every line up to #@REQUESTS@ a comment.
#$ -S /bin/bash
#$ -N @NAME@
#$ -l gpu=@GPUS@
#$ -ac allow=EF
#$ -l h_rt=@TIME@
#$ -wd @TREE@
#$ -o @TREE@/results/sge-$JOB_ID.log
#$ -j y
#@REQUESTS@
