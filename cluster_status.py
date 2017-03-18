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


HEADNODE = "aci-b.aci.ics.psu.edu"
EXCLUDE_HOSTS = [
    'moab-insight01',
    'comp-st-023',
    'comp-st-027',
    'comp-st-035',
    'comp-st-037',
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

        fields_to_get = 2
        for field in all_fields[1:]:
            field = field.strip()
            if field.startswith('np = '):
                host_record['np'] = int(field[5:].strip())
            elif field.startswith('properties = '):
                host_record['properties'] = field[13:].strip().split(',')

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
    time.sleep(10)

    # Do not block...
    for t in threads:
        t.join(0.1)

    while not q.empty():
        host, load = q.get()
        nodes_info[host]['load'] = load

    return nodes_info


def get_load(q, host):
    """Get the 15 min load average from `host` and put on queue `q`"""
    out, err = subprocess.Popen(
        ["ssh", host, "-o StrictHostKeyChecking=no", "cat /proc/loadavg"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    ).communicate()

    load_avg = float(out.split()[2])
    q.put((host, load_avg))


if __name__ == '__main__':
    nodes_info = get_load_info(get_pbsnodes_info())
    normalized_loads = []
    for host, info in nodes_info.iteritems():
        if 'load' not in info:
            print host, info
            continue
        norm_load = info['load'] / info['np']
        normalized_loads.append(norm_load)
        print '%s: np %d, load %3.2f' % (host, info['np'], norm_load)
    print 'Average load across cluster = %s' % np.mean(normalized_loads)
