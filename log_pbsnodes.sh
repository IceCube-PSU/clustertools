#!/bin/bash

mkdir -p /storage/group/dfc13_collab/pbsnodes_logs/

while ((1))
do
	pbsnodes | sed -n '/^comp-cl/,/^\s*$/p' > /storage/group/dfc13_collab/pbsnodes_logs/`isodate`
	sleep 3600
done
