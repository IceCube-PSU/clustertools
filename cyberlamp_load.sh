pbsnodesout=`mktemp`
pbsnodes > $pbsnodesout 2>/dev/null

gpuusage=`cat $pbsnodesout | grep gpu_utilization | tr ';' '\n' | grep gpu_utilization | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
gpumem=`cat $pbsnodesout | grep gpu_memory_used | tr ';' '\n' | grep gpu_memory_used | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
gpumem_tot=`cat $pbsnodesout | grep gpu_memory_total | tr ';' '\n' | grep gpu_memory_total | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
gputemp=`cat $pbsnodesout | grep gpu_utilization | tr ';' '\n' | grep gpu_temp | tr ',' '\n' | grep 'gpu_temp' | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
ngpus=`cat $pbsnodesout | grep gpu_utilization | tr ';' '\n' | grep gpu_utilization -c`
tot_ngpus=`cat $pbsnodesout | grep gpus | tr -s ' ' | cut -d ' ' -f4 | paste -sd+ | bc`
tot_ncpus=`cat $pbsnodesout | grep 'comp-clgc\|comp-clhc' -A 3 | grep np | tr -s ' ' | cut -d ' ' -f4 | paste -sd+ | bc`

nodes_down=`cat $pbsnodesout | grep 'comp-clgc\|comp-clhc' -A 1 | grep "state = down" -c`

cpuusage=`cat $pbsnodesout | grep 'comp-clgc\|comp-clhc' | grep status | tr ',' '\n' | grep loadave | grep -Eo '([0-9]*[.])?[0-9]{1,4}' | paste -sd+ | bc`
ncpus=`cat $pbsnodesout | grep 'comp-clgc\|comp-clhc' | grep status | tr ',' '\n' | grep ncpus | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`

gpus_used_exclusive=`cat $pbsnodesout | grep gpu_status | tr ';' '\n' | grep "gpu_state=Exclusive" -c`
gpus_used_shared=`cat $pbsnodesout | grep gpu_status | tr ';' '\n' | grep "gpu_state=Shared" -c`
gpus_used=$((gpus_used_exclusive + gpus_used_shared))
gpus_free=`cat $pbsnodesout | grep gpu_status | tr ';' '\n' | grep "gpu_state=Unallocated" -c`

cpus_used=`showq -p cyberlamp -r | grep processors | tr -s ' ' | cut -d ' ' -f4`
cpus_free=$((ncpus - cpus_used))


availmem=`cat $pbsnodesout | grep 'comp-clgc\|comp-clhc' | grep status | tr ',' '\n' | grep availmem | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
physmem=`cat $pbsnodesout | grep 'comp-clgc\|comp-clhc' | grep status | tr ',' '\n' | grep physmem | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
av_gpuload=`echo "$gpuusage / $gpus_used" | bc -l`
av_gpumem=`echo "$gpumem / $gpumem_tot / $ngpus * $gpus_used * 100" | bc -l`
av_cpuload=`echo "$cpuusage / $ncpus * 100" | bc -l`
av_temp=`echo "$gputemp / $ngpus" | bc -l`
mem_ratio=`echo "1 - $availmem / $physmem" | bc -l`
av_mem=`echo " $mem_ratio * 100" | bc -l`
cpu_ratio=`echo "$cpus_used / $ncpus * 100" | bc -l` 
gpu_ratio=`echo "$gpus_used / $ngpus * 100" | bc -l` 

rm -f $pbsnodesout >/dev/null 2>&1

printf "%s\n" "--- CyberLAMP CPU summary ---"
printf "CPU cores allocated : %6.2f %% (%s allocated, %s unallocated; %s node(s) down)\n" $cpu_ratio $cpus_used $cpus_free $nodes_down
printf "CPU load            : %6.2f %%\n" $av_cpuload
printf "CPU memory usage    : %6.2f %%\n" $av_mem

printf "\n%s\n" "--- CyberLAMP GPU summary ---"
printf "GPUs allocated      : %6.2f %% (of %s allocated: %s in process-exclusive mode, %s in shared mode; %s unallocated; %s down)\n" $gpu_ratio $(( gpus_used_exclusive + gpus_used_shared )) $gpus_used_exclusive $gpus_used_shared $gpus_free $((tot_ngpus - ngpus))
printf "GPU load            : %6.2f %%\n" $av_gpuload
printf "GPU memory usage    : %6.2f %%\n" $av_gpumem
printf "GPU temperature     : %6.2f degC\n" $av_temp
