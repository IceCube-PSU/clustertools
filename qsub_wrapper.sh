#!/bin/bash

PATH=$PATH:~jpa14/bin

set_real_size=0
if [[ "${1}" == "--real_size" ]]; then
	set_real_size=1
	shift;
fi

if [[ "${1}" == "--group_by" ]]; then
	GROUP_ALL=${2}
	echo "Grouping scripts by ${GROUP_ALL}"
	shift;
	shift;
else
	GROUP_ALL=1
fi

declare -a GROUP_BY

GROUP_BY_LAST=-1
for GROUP in `sed 's/:/ /g' <<<${GROUP_ALL}`;
do
	GROUP_BY_LAST=$(( ${GROUP_BY_LAST} + 1 ))
	if [[ "x${GROUP}" == "x" ]]; then
		continue;
	fi
	GROUP_BY[${GROUP_BY_LAST}]=${GROUP};
done


MEM_MAGNITUDE="gb"

#SUBMITTED_DIR=submitted_${HOSTNAME}

#mkdir ${SUBMITTED_DIR}

files_to_go=`grep -o " " <<<"$*"|wc -l`
files_to_go=$(( $files_to_go + 1 ))

mkdir -p /tmp/${USER}/mergedfiles

function get_info {
	grep "#PBS $1" $2 | sed "s/^#PBS $1//" | sed -r "s/^[ ]*//"
}

declare -a farray

GROUP_I=${GROUP_BY_LAST}

while [[ "x$1" != "x" ]]; do
	GROUP_I=$(( ${GROUP_I} + 1 ))
	if [[ ${GROUP_I} -gt ${GROUP_BY_LAST} ]]; then
		GROUP_I=0
	fi
	n_queued=`qstat -u ${USER} | awk 'BEGIN{ count=0 }{ if($10=="Q") count+=1 }END{ print count }'`
	while [[ $n_queued -gt 100 ]]; do
		echo "Waiting 3 min for queued jobs to be reduced before submitting again."
		date
		#qstat -u ${USER} | awk '{ print $10 }'| sort | uniq -c
		check_queue.sh -q
		showstart `qselect -u ${USER} -s Q | head -n 1` | grep --color=never "start in " | awk '{ print "Next job scheduled to start in "$6 }'
		echo "Number of jobs to be submitted: ${files_to_go}"
		sleep 180
		n_queued=`qstat -u ${USER} | awk 'BEGIN{ count=0 }{ if($10=="Q") count+=1 }END{ print count }'`
	done

	mem=0
	real_size=0
	for i in `seq 0 $(expr ${GROUP_BY[${GROUP_I}]} - 1)`; do
		if [[ "x$1" != "x" ]]; then
			mkdir -p `dirname $1`/merged_submitted
			mv $1 `dirname $1`/merged_submitted
			farray[$i]=$(readlink -e `dirname $1`/merged_submitted/`basename $1`)
			mem_read=`get_info "-l mem=" ${farray[$i]}|sed "s/${MEM_MAGNITUDE}//"`
			mem=`echo "$mem + $mem_read" | bc`
			real_size=$(expr ${real_size} + 1)
			shift;
		else
			farray[$i]=""
		fi
	done
	walltime=`get_info "-l walltime=" ${farray[0]}`
	queue=`get_info "-q" ${farray[0]}`
	log_file=`get_info "-o" ${farray[0]}`
	log_file=`dirname ${log_file}`/merged.`basename ${log_file}`
	name=`get_info "-N" ${farray[0]}`
	mail_opt=`get_info "-m" ${farray[0]}`
	if [[ "x${mail_opt}" == "x" ]]; then
		mail_opt="n"
	fi
	mail_add=`get_info "-M" ${farray[0]}`
	if [[ "x${mail_add}" == "x" ]]; then
		mail_opt="n"
		mail_add=${USER}
	fi
	filename=/tmp/${USER}/mergedfiles/merged.`basename ${farray[0]}`
	if [[ "x${set_real_size}" == "x0" ]]; then
		real_size=${GROUP_BY[${GROUP_I}]}
	fi
	export walltime log_file name queue mem real_size
	touch ${filename}
	cat > ${filename} <<EOF
#PBS -l nodes=1:ppn=${real_size}
#PBS -l mem=${mem}${MEM_MAGNITUDE}
#PBS -l walltime=${walltime}
EOF
	if [[ "x${queue}" != "x" ]]; then
		cat >> ${filename} <<EOF
#PBS -q ${queue}
EOF
	fi
	cat >> ${filename} <<EOF
#PBS -j oe
#PBS -o ${log_file}
#PBS -N ${name}
#PBS -m ${mail_opt}
#PBS -M ${mail_add}

echo "Job \${PBS_JOBNAME} running on \${HOSTNAME}"

function bash_run {
	(bash \$1 && rm -f \$1) &
}

EOF
	for i in `seq 0 $(expr ${GROUP_BY[${GROUP_I}]} - 1 )`; do
		if [[ -e ${farray[$i]} ]]; then
			local_logging_file=`get_info "-o" ${farray[$i]}`
			cat >> ${filename} <<EOF
bash_run ${farray[$i]} > ${local_logging_file} 2>&1
EOF
		fi
	done
	cat >> ${filename} <<EOF
#wait
for job in \`jobs -p\`;
do
	wait \$job;
done

# vim: syntax=sh
EOF

	echo "Submitting file ${filename}"
	qsub ${filename}
	status=$?
	while [[ "x$status" != "x0" ]]; do
		echo "Submission did not work. Waiting 5 min before trying again."
		echo "Number of jobs to be submitted: ${files_to_go}"
		sleep 300
		qsub ${filename}
		status=$?
	done
	rm ${filename}
	echo "====================================================="

	files_to_go=`grep -o " " <<<"$*"|wc -l`
done
