#!/usr/bin/env python

from copy import copy
import re
import subprocess

import pandas as pd


FIELDS = [
    'physical id',
    'core id',
    'processor',
]

NAME_VAL_RE = re.compile(r'^(\S.+\S)\t*:[ ]*(\S.*){0,1}$')
VALUE_RE = re.compile(r'\t*:[ ]*(\S.*)$')


def parseProcSection(txt, fields=FIELDS):
    out = {}
    fields = list(copy(fields))
    for line in txt.split('\n'):
        match = NAME_VAL_RE.match(line)
        if match is None:
            print ('WARNING: Could not process line:\n' + line)
            continue
        name, val = match.groups()
        if name not in fields:
            continue
        fields.remove(name)
        out[name] = int(val)
        if len(fields) == 0:
            break
    return out

def getCPULayout():
    rawtxt = subprocess.check_output(["cat", "/proc/cpuinfo"])
    processor_sections = (rawtxt).strip().split('\n\n')
    df = pd.DataFrame([parseProcSection(s) for s in processor_sections],
                      columns=FIELDS)
    df.sort_values(FIELDS, inplace=True)
    df = pd.DataFrame(df['processor'].values,
                      index=[df['physical id'], df['core id']],
                      columns=['processor'])
    df = df.rename_axis(['physical_id', 'core_id'])
    return df

if __name__ == '__main__':
    df = getCPULayout()
    print '%11s  %7s  %8s' %('physical id', 'core id', '"cpu(s)"')
    previous_physical_id = None
    for n, idx in enumerate(df.index.unique()):
        physical_id, core_id = idx
        if physical_id != previous_physical_id:
            print '%11s  %7s  %8s' %('-'*11, '-'*7, '-'*8)
            previous_physical_id = physical_id
        print '%11s  %7s  %8s' %(
            physical_id, core_id,
            ', '.join(['%2s' %p for p in sorted(df.ix[idx]['processor'].values)])
        )
