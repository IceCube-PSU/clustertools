i=1
qselect -u $USER $1 | while read x; do qdel $x & if (( i%100==0 )); then wait; fi; i=$(( i+1 )); done
