#!/bin/bash

mkdir -p /storage/group/dfc13_collab/pbsnodes_logs/

while ((1))
do
	pbsnodes > /tmp/pbsnodes
	lines=`grep -e '^comp-cl'  -n /tmp/pbsnodes | head -1 | awk -F':' '{print $1}'`
	tail --lines=+$lines /tmp/pbsnodes > /storage/group/dfc13_collab/pbsnodes_logs/`isodate`
	sleep 3600
done
