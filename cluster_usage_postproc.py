#!/usr/bin/env python
"""
Obtain detailed usage information for the ICS-ACI cluster
"""


# TODO: Kill errant thread(s)
# TODO: Other interesting things...


from argparse import ArgumentParser
from collections import OrderedDict
from copy import copy, deepcopy
import os
import re
import subprocess
import threading
import time

import dateutil.parser
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from genericUtils import mkdir, timestamp
from plotGoodies import removeBorder



def parse_args(description=__doc__):
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '-d', metavar='DIR', required=True,
        help='Log dir'
    )
    return parser.parse_args()


FILENAME_DATETIME_RE = re.compile(r'cluster_usage.(.+)')
def get_datetime(filepath):
    _, filename = os.path.split(filepath)
    base, _ = os.path.splitext(filename)
    dt_str = FILENAME_DATETIME_RE.findall(base)[0]
    return dateutil.parser.parse(dt_str)

NODE_INFO_RE = re.compile(
    r'^comp-(?P<nodetype>st|hm)-(?P<nodenum>[0-9])+ .*'
    r'cores = (?P<cores>[0-9]+), mem = *(?P<mem>[0-9]+) .*\[ *'
    r'(?P<load>[0-9]+) *(?P<memusage>[0-9]+)'
)


def get_info(args):
    args.d = os.path.expandvars(os.path.expanduser(args.d))
    records = []
    for filename in os.listdir(args.d):
        filepath = os.path.join(args.d, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            dt = get_datetime(filepath)
        except Exception, err:
            print err
            continue

        with file(os.path.join(args.d, filepath), 'r') as f:
            contents = f.readlines()

        for line in contents:
            if line[0] == '[':
                if line.strip() == '[failed_to_report]':
                    break
                else:
                    continue
            info = dict(dt=dt)
            match = NODE_INFO_RE.match(line)
            if match is None:
                continue
            info.update(match.groupdict())
            info['cores'] = int(info['cores'])
            info['load'] = float(info['load'])/100
            info['mem'] = float(info['mem'])
            info['memusage'] = float(info['memusage'])/100
            info['nodenum'] = int(info['nodenum'])
            records.append(info)

    records = pd.DataFrame(records)
    records.sort_values(by=['dt', 'nodetype', 'nodenum'], inplace=True)
    return records


def main():
    """main"""
    records = get_info(parse_args())
    slot_mem = records['mem'] / records['cores']
    cores_avail = np.clip(
        records['cores'] * (1 - records['load']),
        a_min=0, a_max=np.inf
    )
    mem_avail = np.clip(
        records['mem'] * (1 - records['memusage']),
        a_min=0, a_max=np.inf
    )
    records['slots_avail'] = np.min([cores_avail, mem_avail//slot_mem],
                                    axis=0)

    summary = []
    for dt, dt_group in records.groupby('dt'):
        for nodetype, nt_group in dt_group.groupby('nodetype'):
            capacity = np.sum(nt_group['cores'])
            available = np.sum(nt_group['slots_avail'])
            used = capacity - available
            summary.append({
                'dt': dt,
                'nodetype': nodetype,
                'percent used': used / capacity*100,
                #'slots used': used,
                #'reported capacity': capacity
            })
    summary = pd.DataFrame(summary)
    summary.set_index('dt', inplace=True)

    for nt, nt_group in summary.groupby('nodetype'):
        ax = nt_group.plot(
            title='High Mem' if nt == 'hm' else 'Standard Mem'
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('Slots')
        removeBorder(ax)
        ax.set_ylim(0, 100) #ax.get_ylim()[1])
        xlims = ax.get_xlim()
        #ax.plot(xlims, [
        plt.savefig(nt + '.png', dpi=300)
        plt.savefig(nt + '.pdf')
    plt.draw()
    plt.show()

    return records, summary


if __name__ == '__main__':
    records, summary = main()
