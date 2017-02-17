#!/bin/bash

ANALYSIS="llr"
N_TRIALS_PER_PROC=$1
N_GPUS_TO_USE=$2
N_PROCS_PER_GPU=$3
TEMPLATE_SETTINGS=$4
MINIMIZER_SETTINGS=$5
#EXTRA_LABEL=$6
#CPU_AFFINITY=$5

N_GPUS=`nvidia-smi --list-gpus | wc -l`

# Determine the information to populate below tables via following commands:
# 1. Get the GPU numbers you want to use on the system.
#    Note that we had issues determining without trying things out which gpu
#    number (i.e., what works for export CUDA_VISIBLE_DEVICES=<gpu number>)
#    corresponded to the two K40s on schwyz, which also has a GT 610 which
#    possibly messes things up. Turns out 0 and 2 get you to what appear in
#    nvidia-smi as 1 and 2 (the two K40 gpus).
#
#    nvidia-smi --query-gpu=index,name,uuid,serial --format=csv
#
# 2. Get CPU affinity for each GPU
#
#    nvidia-smi topo --matrix
#
# 3. Figure out which cpu numbers to assign to each GPU. Make sure that two CPU
#    numbers aren't hyperthread versions of the same core. You want to record
#    the "processor" number, making sure you don't specify in the list two
#    processor numbers with the same "core id" (the line immediately below
#    "processor" is the "core id" corresponding to that "processor").
#    
#    cat /proc/cpuinfo  | grep processor -A 11 | grep -E "^(processor|physical id|core id)"
#
#    Or better yet, run script from the psu-icecube Github clustertools repo:
#
#    clustertools/pprint_cpu_layout.py
#
# 4. Populate the variables below:
#    GPUS : list of GPU numbers
#    CPUS[<gpu number>] : list of processor numbers usable by <gpu number>

declare -A CPUS
if [ "$HOSTNAME" == "schwyz" ]
then
	GPUS="0 2"
	CPUS[0]=" 0  1  2  3  4  5  6  7"
	CPUS[2]=" 8  9 10 11 12 13 14 15"
elif [ "$HOSTNAME" == "uri" ]
then
	GPUS="0 1 2 3"
	CPUS[0]=" 0  1  2  3"
	CPUS[1]=" 4  5  6  7"
	CPUS[2]=" 8  9 10 11"
	CPUS[3]="12 13 14 15"
elif [ "$HOSTNAME" == "unterwalden" ]
then
	GPUS="0 1 2 3"
	CPUS[0]=" 0  1"
	CPUS[1]=" 2  3"
	CPUS[2]=" 4  5"
	CPUS[3]=" 6  7"
fi

SCRIPT="$PISA/pisa/analysis/llr/LLROptimizerAnalysis.py"

BASE_TS="`basename $TEMPLATE_SETTINGS | sed 's/\.[^.]*$//'`"
BASE_MS="`basename $MINIMIZER_SETTINGS | sed 's/\.[^.]*$//'`"
OUTDIR="/fastio/pingu_analysis"
TASKNAME="${ANALYSIS}__${BASE_TS}__${BASE_MS}"
RUNID="`date -u +%Y-%m-%dT%H%M%S`__${HOSTNAME}"
BASEDIR="${OUTDIR}/${TASKNAME}"
OUTDIR="${BASEDIR}/results_rawfiles"
LOGDIR="${BASEDIR}/logfiles"

mkdir -p $OUTDIR
mkdir -p $LOGDIR

echo "> N_GPUS=$N_GPUS; N_GPUS_TO_USE=$N_GPUS_TO_USE"

#for ((gpu_num=N_GPUS-1; gpu_num >= N_GPUS-N_GPUS_TO_USE; gpu_num--))
for gpu_num in $GPUS
do
	cpus_for_this_gpu=${CPUS[$gpu_num]}
	gpu_subrpoc_num=0
	for cpu in $cpus_for_this_gpu
	do
		# Try to make MKL only use one thread
		export OMP_NUM_THREADS=1

		# Set CPU affinity on machines where Intek MKL takes over all available
		# CPUs (and doesn't run any faster, at least for PISA)
		set_affinity="taskset -c $cpu"

		# Set GPU affinity
		export CUDA_VISIBLE_DEVICES=$gpu_num

		FULLJOBID="${TASKNAME}__${RUNID}__G${gpu_num}.${gpu_subrpoc_num}"

		# Report what has been established
		echo "-> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, cpu=$cpu;" \
				" N_PROCS_PER_GPU=$N_PROCS_PER_GPU," \
				" N_TRIALS_PER_PROC=$N_TRIALS_PER_PROC"
		echo "   " "$OUTSUB"
		echo "   " "$FULLJOBID"

		# Launch the multi-at-once script
		COMMAND='$set_affinity "$SCRIPT" --save-steps --template-settings "$TEMPLATE_SETTINGS" --minimizer-settings "$MINIMIZER_SETTINGS" --ntrials $N_TRIALS_PER_PROC --outfile "${OUTDIR}/${FULLJOBID}.json" > "${LOGDIR}/${FULLJOBID}.out" 2> "${LOGDIR}/${FULLJOBID}.err"'
		echo "$SCRIPT"
		echo "$COMMAND"
		eval $COMMAND &
		echo ""
		gpu_subrpoc_num=$(( gpu_subrpoc_num + 1 ))
		if (( gpu_subrpoc_num == N_PROCS_PER_GPU ))
		then
			break
		fi
	done
done
