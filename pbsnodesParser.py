import re
import sys
sys.path.append('./lib')

from vsc.pbs.pbsnodes import PbsnodesParser
#from vsc.pbs.node import NodeStatus

parser = PbsnodesParser()

#
# Next incarnation of this script will do the following:
# - loop over all accumulated pbsnodes files
#   - extract time from filename
# - keep track of various params as a function of node type
#   - node type: cyberlamp hi-mem, aci standard,...
#   - params: number of free cpus, number of sessions, other things
# - plot everything vs. time
#   - add average of each quantity to plot title

pbs_file = open("./pbsnodes.dat",'r')
nodes = parser.parse_file(pbs_file)

total_cpu = 0
total_sessions = 0

for node in nodes:
    if re.match(r'comp-clgc-0001',node.hostname):
#    if re.match(r'comp-cl',node.hostname):

        this_node_mem = 0

        print node.hostname,node.np,node.memory
        total_cpu += int(node.status['ncpus'])
        total_sessions += int(node.status['nsessions'])
        pmem = int(re.split(r'\D+',node.status['physmem'])[0])
        totmem = int(re.split(r'\D+',node.status['totmem'])[0])
        availmem = int(re.split(r'\D+',node.status['availmem'])[0])
#        print node.memory
#        print node.memload
        number_of_jobs = len(node.jobs.keys())
        for jobnum in node.jobs.keys():
            full_jobnum = node.jobs[jobnum]
#           Possible keys: ['energy_used', 'mem', 'cput', 'session_id', 'vmem', 'walltime']
            print node.status['job_info'][full_jobnum]['mem']
            job_mem = int(re.split(r'\D+',node.status['job_info'][full_jobnum]['mem'])[0])
            this_node_mem += job_mem

#        print node.jobs.keys()
#        print node.job_ids

#        print node.status['job_info']
#        print node.status.keys()

#
# Naively expected that total memory minus available memory would
# equal memory used, i.e., "this_node_mem" but it does not...
#
print "totmem-availmem, this_node_mem: ",totmem-availmem,this_node_mem
print "pmem-availmem, this_node_mem: ",pmem-availmem,this_node_mem

print number_of_jobs
print total_sessions
print total_cpu


#            print node.status['job_info'][full_jobnum]['session_id']

#        print node.properties
#        node's 1 min bsd load average (number of processes wanted to run in last min)
#        print node.status['loadave'] 
