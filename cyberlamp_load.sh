gpuusage=`pbsnodes | grep gpu_utilization | tr ';' '\n' | grep gpu_utilization | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
gpumem=`pbsnodes | grep gpu_memory_used | tr ';' '\n' | grep gpu_memory_used | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
gpumem_tot=`pbsnodes | grep gpu_memory_total | tr ';' '\n' | grep gpu_memory_total | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
gputemp=`pbsnodes | grep gpu_utilization | tr ';' '\n' | grep gpu_temp | tr ',' '\n' | grep 'gpu_temp' | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
ngpus=`pbsnodes | grep gpu_utilization | tr ';' '\n' | grep gpu_utilization -c`

cpuusage=`pbsnodes | grep 'comp-clgc\|comp-clhc' | grep status | tr ',' '\n' | grep loadave | grep -Eo '([0-9]*[.])?[0-9]{1,4}' | paste -sd+ | bc`
ncpus=`pbsnodes | grep 'comp-clgc\|comp-clhc' | grep status | tr ',' '\n' | grep ncpus | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`

gpus_used=`pbsnodes | grep gpu_status | tr ';' '\n' | grep "gpu_state=Exclusive" -c`
gpus_free=`pbsnodes | grep gpu_status | tr ';' '\n' | grep "gpu_state=Unallocated" -c`

cpus_used=`showq -p cyberlamp -r | grep processors | tr -s ' ' | cut -d ' ' -f4`
cpus_free=$((ncpus - cpus_used))


availmem=`pbsnodes | grep 'comp-clgc\|comp-clhc' | grep status | tr ',' '\n' | grep availmem | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
physmem=`pbsnodes | grep 'comp-clgc\|comp-clhc' | grep status | tr ',' '\n' | grep physmem | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
av_gpuload=`echo "$gpuusage / $gpus_used" | bc -l`
av_gpumem=`echo "$gpumem / $gpumem_tot * $ngpus / $gpus_used * 100" | bc -l`
av_cpuload=`echo "$cpuusage / $cpus_used * 100" | bc -l`
av_temp=`echo "$gputemp / $ngpus" | bc -l`
mem_ratio=`echo "1 - $availmem / $physmem" | bc -l`
av_mem=`echo " $mem_ratio * 100" | bc -l`
cpu_ratio=`echo "$cpus_used / $ncpus * 100" | bc -l` 
gpu_ratio=`echo "$gpus_used / $ngpus * 100" | bc -l` 

printf "%s\n" "--- CPU summary ---"
printf "cluster CPU usage        : %.2f %% (%s cores in use; %s cores free)\n" $cpu_ratio $cpus_used $cpus_free
printf "average CPU load         : %.2f %%\n" $av_cpuload
printf "total CPU memory usage   : %.2f %%\n" $av_mem
printf "\n%s\n" "--- GPU summary ---"
printf "cluster GPU usage        : %.2f %% (%s GPUs in use; %s GPUs free)\n" $gpu_ratio $gpus_used $gpus_free
printf "average GPU load         : %.2f %%\n" $av_gpuload
printf "average GPU memory usage : %.2f %%\n" $av_gpumem
printf "average GPU temperature  : %.2f degC\n" $av_temp
