#!/usr/bin/env python
"""
Obtain detailed usage information for the ICS-ACI cluster
"""


# TODO: Kill errant thread(s)
# TODO: Other interesting things...


from argparse import ArgumentParser
from collections import OrderedDict
from copy import copy
import os
import Queue
import subprocess
import threading
import time

import numpy as np

from genericUtils import mkdir, timestamp


CORES_PER_JOB = 1
MEM_PER_JOB_GB = 4

RESERVED_CORES = 1
RESERVED_MEM = 4


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


def get_load_info(nodes_info, timeout):
    q = Queue.Queue()
    threads = []
    for host in nodes_info.keys():
        t = threading.Thread(target=get_load, args=(q, host))
        t.daemon = True
        threads.append(t)
        t.start()

    t0 = time.time()
    while time.time() < t0 + timeout:
        for t in copy(threads):
            if not t.isAlive():
                threads.remove(t)
        if len(threads) == 0:
            break
        time.sleep(0.2)

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
        return

    try:
        fields = out.split('\n')
        load_avg = float(fields[0].split(' ')[2])
        mem_total = float(fields[1].replace('MemTotal:', '').replace('kB', '').strip())/1024/1024
        mem_free = float(fields[2].replace('MemFree:', '').replace('kB', '').strip())/1024/1024
    except ValueError:
        pass
        #print '='*80
        #print 'Bad or no output from host:', host
        #print 'out ='
        #print out
        #print '-'*80
        #print 'err ='
        #print err
        #print '='*80
    else:
        q.put((host, load_avg, mem_total, mem_free))


def parse_args(description=__doc__):
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '-l', action='store_true',
        help='Whether to log full info'
    )
    parser.add_argument(
        '-d', metavar='DIR', default='',
        help='Log dir; otherwise log to PWD'
    )
    parser.add_argument(
        '-t', metavar='SEC', default=60, type=float,
        help='Timeout for receiving responses, in seconds'
    )
    parser.add_argument(
        '-v', action='store_true',
        help='Print all collected info to stdout'
    )
    return parser.parse_args()


def main(args):
    ts = timestamp(winsafe=True)
    if args.l or args.v:
        record_all_info = True

    if args.d:
        mkdir(args.d, warn=False)

    nodes_info = get_load_info(nodes_info=get_pbsnodes_info(),
                               timeout=args.t)
    normalized_loads = []
    mem_used_fracts = []
    unable_to_get = []
    full_by_mem_usage = []
    full_by_cpu_usage = []
    slots_available = []
    strings = []

    strings.append('[per_host_info]')

    for host, info in nodes_info.iteritems():
        if 'load' not in info:
            unable_to_get.append(
                '%-15s: cores = %2d, failure = could not get info'
                % (host, info['np'])
            )
            continue

        norm_load = info['load'] / info['np']
        normalized_loads.append(norm_load)

        mem_used = info['mem_total'] - info['mem_free']
        mem_used_fract = mem_used / info['mem_total']
        mem_used_fracts.append(mem_used_fract)

        effective_cores_available = np.floor(info['np'] - info['load'])
        if effective_cores_available < RESERVED_CORES + CORES_PER_JOB:
            f_by_c = True
        else:
            f_by_c = False
        full_by_cpu_usage.append(f_by_c)

        if info['mem_free'] < RESERVED_MEM:
            f_by_m = True
        else:
            f_by_m = False
        full_by_mem_usage.append(f_by_m)

        if 'comp-st' in host:
            full = f_by_m or f_by_c
            mem_slots_avail = (
                int(not full)
                * (info['mem_free'] - RESERVED_MEM) // MEM_PER_JOB_GB
            )
            cpu_slots_avail = (
                int(not full) * (
                    effective_cores_available - RESERVED_CORES
                ) // CORES_PER_JOB
            )
            slots_available.append(min(mem_slots_avail, cpu_slots_avail))

        # TODO: make -v command-line option and output the following
        strings.append(
            '%-15s: cores = %2d, mem = %4d GB, load_mem_pct = [ %3d  %3d ]'
            % (host, info['np'], info['mem_total'],
               np.round(norm_load*100), np.round(mem_used_fract*100))
        )

    strings.append('')
    strings.append('[failed_to_report]')
    for s in unable_to_get:
        strings.append(s)

    if args.l:
        fpath = os.path.expandvars(os.path.expanduser(
            os.path.join(args.d, 'cluster_usage.%s.log' % ts)
        ))
        with file(fpath, 'w') as f:
            for s in strings:
                f.write(s + '\n')
        print 'Wrote %d lines to "%s"' % (len(strings), fpath)
        print ''

    if args.v:
        for s in strings:
            print s
        print ''
        print '---- Summary ----'
        print ''

    avg_load = np.mean(normalized_loads)
    avg_mem_used = np.mean(mem_used_fracts)

    print('Average load/mem usage across cluster = [ %3d  %3d ]%s'
          % (np.round(100*avg_load), np.round(100*avg_mem_used), '%'))

    full_by_mem_usage = np.array(full_by_mem_usage)
    full_by_cpu_usage = np.array(full_by_cpu_usage)

    total_nodes_found = len(full_by_mem_usage)
    total_full_by_cpu = np.sum(full_by_cpu_usage & ~full_by_mem_usage)
    total_full_by_mem = np.sum(full_by_mem_usage & ~full_by_cpu_usage)
    total_full_by_both = np.sum(full_by_cpu_usage & full_by_mem_usage)
    total_full_by_either = np.sum(full_by_cpu_usage | full_by_mem_usage)

    print(
        'Of %3d nodes reporting, %3d are full: %3d by CPU, %3d by mem, and %3d'
        ' by both.'
        % (total_nodes_found, total_full_by_either, total_full_by_cpu,
           total_full_by_mem, total_full_by_both)
    )
    print(
        '  Note: "full" if: <= %d cores available or < %d GB available.'
        % (RESERVED_CORES, RESERVED_MEM+MEM_PER_JOB_GB)
    )

    print(
        'Of %3d standard-mem nodes reporting, there are %3d slots available'
        ' for %d core / %d GB jobs.'
        % (len(slots_available), np.sum(slots_available), CORES_PER_JOB,
           MEM_PER_JOB_GB)
    )
    print (
        '(Unable to get info from %d nodes)'
        % (len(unable_to_get) + len(EXCLUDE_HOSTS) - 1)
    )


if __name__ == '__main__':
    main(parse_args())
