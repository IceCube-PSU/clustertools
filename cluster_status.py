#!/usr/bin/env python


# TODO: Kill errant thread(s)
# TODO: Get memory info for each node to determine if it is memory-limited
# TODO: Cache results to shared folder with a timestamp so we don't have to
#       query all hosts to get this info... or separate the "get info" and
#       "report info" into separate scripts, where one user runs via cron
#       the "get info" part regularly, and everyone else simply reads the
#       file produced by that.
# TODO: Other interesting things...


from collections import OrderedDict
import Queue
import subprocess
import threading
import time

import numpy as np
import pandas as pd


MIN_CORES_TO_RUN_JOB = 2
MIN_MEM_TO_RUN_JOB_GB = 8
CORES_PER_JOB = 1
MEM_PER_JOB_GB = 4


HEADNODE = "aci-b.aci.ics.psu.edu"
EXCLUDE_HOSTS = [
    'moab-insight01',
    'comp-st-023',
    'comp-st-027',
    'comp-st-035',
    'comp-st-037',
    'comp-st-222',
    'comp-hm-03',
]


def get_pbsnodes_info():
    out, err = subprocess.Popen(
        ["pbsnodes"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    ).communicate()

    nodes_info = OrderedDict()

    host_sections = out.split('\n\n')
    for host_section in host_sections:
        all_fields = host_section.splitlines()
        if len(all_fields) == 0:
            continue

        host_record = OrderedDict()
        host = all_fields[0].strip()
        if host in EXCLUDE_HOSTS:
            continue

        fields_to_get = 1
        for field in all_fields[1:]:
            field = field.strip()
            if field.startswith('np = '):
                host_record['np'] = int(field[5:].strip())
            #elif field.startswith('properties = '):
            #    host_record['properties'] = field[13:].strip().split(',')

            if len(host_record) == fields_to_get:
                break

        nodes_info[host] = host_record

    return nodes_info


def get_load_info(nodes_info):
    q = Queue.Queue()
    threads = []
    for host in nodes_info.keys():
        t = threading.Thread(target=get_load, args=(q, host))
        t.daemon = True
        threads.append(t)
        t.start()

    # Give some time for commands to complete; when we figure out
    # how to kill errant threads, we can implement a more subtle
    # version of this that probably will take less time.
    time.sleep(15)

    # Do not block...
    for t in threads:
        t.join(0.1)

    while not q.empty():
        host, load, mem_total, mem_free = q.get()
        nodes_info[host]['load'] = load
        nodes_info[host]['mem_total'] = mem_total
        nodes_info[host]['mem_free'] = mem_free

    return nodes_info


def get_load(q, host):
    """Get the 15 min load average from `host` and put on queue `q`"""
    out, err = subprocess.Popen(
        ["ssh", host, "-o StrictHostKeyChecking=no",
         "cat /proc/loadavg ; cat /proc/meminfo | grep \"Mem\""],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    ).communicate()

    if 'connection closed' in err.lower():
        print 'Connection closed by host:', host
        return

    try:
        fields = out.split('\n')
        load_avg = float(fields[0].split(' ')[2])
        mem_total = float(fields[1].replace('MemTotal:', '').replace('kB', '').strip())/1024/1024
        mem_free = float(fields[2].replace('MemFree:', '').replace('kB', '').strip())/1024/1024
    except ValueError:
        print '='*80
        print 'Bad or no output from host:', host
        print 'out ='
        print out
        print '-'*80
        print 'err ='
        print err
        print '='*80
    else:
        q.put((host, load_avg, mem_total, mem_free))


if __name__ == '__main__':
    nodes_info = get_load_info(get_pbsnodes_info())
    normalized_loads = []
    mem_used_fracts = []
    unable_to_get = []
    full_by_mem_usage = []
    full_by_cpu_usage = []
    slots_available = []

    for host, info in nodes_info.iteritems():
        if 'load' not in info:
            unable_to_get.append(
                '%-15s: %2d cores, --> unable to get load & memory info.'
                % (host, info['np'])
            )
            continue

        norm_load = info['load'] / info['np']
        normalized_loads.append(norm_load)

        mem_used = info['mem_total'] - info['mem_free']
        mem_used_fract = mem_used / info['mem_total']
        mem_used_fracts.append(mem_used_fract)

        effective_cores_available = np.floor(info['np'] - info['load'])
        if effective_cores_available < MIN_CORES_TO_RUN_JOB:
            f_by_c = True
        else:
            f_by_c = False
        full_by_cpu_usage.append(f_by_c)

        if info['mem_free'] < MIN_MEM_TO_RUN_JOB_GB:
            f_by_m = True
        else:
            f_by_m = False
        full_by_mem_usage.append(f_by_m)

        if 'comp-st' in host:
            full = f_by_m or f_by_c
            mem_slots_avail = int(not full) * info['mem_free'] // MEM_PER_JOB_GB
            cpu_slots_avail = int(not full) * effective_cores_available // CORES_PER_JOB
            slots_available.append(min(mem_slots_avail, cpu_slots_avail))

        # TODO: make -v command-line option and output the following
        #print (
        #    '%-15s: %2d cores, %4d GB: load/mem = [ %3d  %3d ]%s' % (
        #        host,
        #        info['np'],
        #        info['mem_total'],
        #        np.round(norm_load*100),
        #        np.round(mem_used_fract*100),
        #        '%',
        #    )
        #)
    avg_load = np.mean(normalized_loads)
    avg_mem_used = np.mean(mem_used_fracts)

    print ''
    print ('**     Average load/mem usage across cluster = [ %3d  %3d ]%s **'
           % (np.round(100*avg_load), np.round(100*avg_mem_used), '%'))

    full_by_mem_usage = np.array(full_by_mem_usage)
    full_by_cpu_usage = np.array(full_by_cpu_usage)

    total_nodes_found = len(full_by_mem_usage)
    total_full_by_cpu = np.sum(full_by_cpu_usage & ~full_by_mem_usage)
    total_full_by_mem = np.sum(full_by_mem_usage & ~full_by_cpu_usage)
    total_full_by_both = np.sum(full_by_cpu_usage & full_by_mem_usage)
    total_full_by_either = np.sum(full_by_cpu_usage | full_by_mem_usage)

    print ''
    print (
        'Of %3d nodes, %3d are full: %3d by CPU, %3d by mem, and %3d by both.'
        % (total_nodes_found, total_full_by_either, total_full_by_cpu,
           total_full_by_mem, total_full_by_both)
    )
    print (
        '  Note: "full" criteria: fewer than %d cores -or- %d GB available.'
        % (MIN_CORES_TO_RUN_JOB, MIN_MEM_TO_RUN_JOB_GB)
    )

    print ''
    print (
        'Of %3d standard-mem nodes reporting, there are %3d slots available for'
        ' %d core / %d GB jobs.'
        % (len(slots_available), np.sum(slots_available), CORES_PER_JOB,
           MEM_PER_JOB_GB)
    )

    # TODO: make -v command-line option and output the following
    #print ''
    #for s in unable_to_get:
    #    print s
