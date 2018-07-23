i=0
jobs=`qselect -u $USER`
echo $jobs $1 | while read x; do qdel $x & if (( i%100==0 )); then wait; fi; i=$(( i+1 )); done
