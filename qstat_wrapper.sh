#!/bin/sh

PATH=$PATH:~jpa14/bin

n_job=`qstat -u ${USER} | awk 'BEGIN{ count=0 }{ if($10=="Q"||$10=="R") count+=1 }END{ print count }'`
while [[ $n_job -ge 1 ]]; do
	echo "Waiting 5 min before probing again number of jobs."
	date
	#qstat -u ${USER} | awk '{ print $10 }'| sort | uniq -c
	check_queue.sh -q
	JOB_ID=`qselect -u ${USER} -s Q | head -n 1`
	if [[ "${JOB_ID}" != "" ]];
	then
		showstart ${JOB_ID} | grep --color=never "start in " | awk '{ print "Next job scheduled to start in "$6 }'
	fi
	sleep 300
	n_job=`qstat -u ${USER} | awk 'BEGIN{ count=0 }{ if($10=="Q"||$10=="R") count+=1 }END{ print count }'`
done
