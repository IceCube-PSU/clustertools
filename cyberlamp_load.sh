gpuusage=`pbsnodes | grep gpu_utilization | tr ';' '\n' | grep gpu_utilization | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
gpumem=`pbsnodes | grep gpu_memory_utilization | tr ';' '\n' | grep gpu_memory_utilization | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
gputemp=`pbsnodes | grep gpu_utilization | tr ';' '\n' | grep gpu_temp | tr ',' '\n' | grep 'gpu_temp' | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
ngpus=`pbsnodes | grep gpu_utilization | tr ';' '\n' | grep gpu_utilization -c`

cpuusage=`pbsnodes | grep 'comp-clgc\|comp-clhc' | grep status | tr ',' '\n' | grep loadave | grep -Eo '([0-9]*[.])?[0-9]{1,4}' | paste -sd+ | bc`
ncpus=`pbsnodes | grep 'comp-clgc\|comp-clhc' | grep status | tr ',' '\n' | grep ncpus | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`

availmem=`pbsnodes | grep 'comp-clgc\|comp-clhc' | grep status | tr ',' '\n' | grep availmem | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
physmem=`pbsnodes | grep 'comp-clgc\|comp-clhc' | grep status | tr ',' '\n' | grep physmem | grep -Eo '[0-9]{1,4}' | paste -sd+ | bc`
av_gpuload=`echo "$gpuusage / $ngpus" | bc -l`
av_gpumem=`echo "$gpumem / $ngpus" | bc -l`
av_cpuload=`echo "$cpuusage / $ncpus * 100" | bc -l`
av_temp=`echo "$gputemp / $ngpus" | bc -l`
mem_ratio=`echo "1 - $availmem / $physmem" | bc -l`
av_mem=`echo " $mem_ratio * 100" | bc -l`

printf "average CPU load         : %.2f %%\n" $av_cpuload
printf "average CPU memory usage : %.2f %%\n" $av_mem
printf "average GPU load         : %.2f %%\n" $av_gpuload
printf "average GPU memory usage : %.2f %%\n" $av_gpumem
printf "average GPU temperature  : %.2f degC\n" $av_temp
