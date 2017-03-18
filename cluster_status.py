#!/usr/bin/env python

from collections import OrderedDict
import Queue
import subprocess
import threading


HEADNODE = "aci-b.aci.ics.psu.edu"
EXCLUDE_HOSTS = ['moab-insight01']


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

    for t in threads:
        t.join()

    while not q.empty():
        host, load = q.get()
        nodes_info[host]['load'] = load

    return nodes_info


def get_load(q, host):
    """Get the 15 min load average from `host` and put on queue `q`"""
    out, err = subprocess.Popen(
        ["ssh", host, "cat /proc/loadavg"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    ).communicate()

    load_avg = float(out.split()[3])
    q.put((host, load_avg))


if __name__ == '__main__':
    nodes_info = get_load_info(get_pbsnodes_info())
