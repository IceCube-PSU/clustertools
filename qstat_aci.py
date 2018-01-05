#!/usr/bin/env python


"""
Parse qstat output and show pertinent info summary for ACI and CyberLAMP
clusters (e.g. on which sub-queues jobs are actually running).
"""


from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from getpass import getuser
from gzip import GzipFile
from math import ceil
from os import makedirs
from os.path import expanduser, expandvars, getmtime, isfile, join
import re
from subprocess import check_output
from time import time
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from pd_utils import convert_df_dtypes


# TODO: get openQ status! e.g. check how many files are in all users' openQ
# folders, as well as how many of those are my files. Can also do something
# with qstat's report of jobs that I have running on ACI open, though would
# need to coordinate with other users to get their info via qstat (i.e. if
# we don't use showq)

# TODO: fix the cached output / read from/to hdf5
# TODO: use the cached qstat hdf5 (or other) file from each user to get group info
# TODO: function like get_jobs but for openq info (number of files in users'
#       dirs; which users' openQ daemons are running, ... anything else?)
# TODO: remove dependencies as much as possible, make pandas optional for at
#       least parts of the script that we can run from OpenQ clients


__all__ = ['USER', 'SORT_COLS', 'QSTAT_CACHE_DIR', 'STALE_TIME',
           'CYBERLAMP_QUEUES', 'ACI_QUEUES', 'CYBERLAMP_CLUSTER_QUEUES',
           'ACI_CLUSTER_QUEUES', 'OPENQ_CLUSTER_QUEUES',
           'KNOWN_CLUSTER_QUEUES', 'ARRAY_RE', 'get_xml_val', 'convert_size',
           'get_qstat_output', 'get_jobs']


pd.set_option('display.max_rows', None)


def expand(p):
    return expanduser(expandvars(p))


USER = getuser()
SORT_COLS = ['cluster', 'queue', 'job_state', 'job_id']
QSTAT_CACHE_DIR = expand('/gpfs/group/dfc13/default/qstat_out')
STALE_TIME = 120 # seconds

CYBERLAMP_QUEUES = ['default', 'cl_open', 'cl_gpu', 'cl_higpu', 'cl_himem',
                    'cl_debug', 'cl_phi']
ACI_QUEUES = ['dfc13_a_g_sc_default', 'dfc13_a_t_bc_default', 'open']
OPENQ_QUEUES = ['openq']

CYBERLAMP_CLUSTER_QUEUES = [('cyberlamp', q) for q in CYBERLAMP_QUEUES]
ACI_CLUSTER_QUEUES = [('aci', q) for q in ACI_QUEUES]
OPENQ_CLUSTER_QUEUES = [(q, q) for q in OPENQ_QUEUES]
KNOWN_CLUSTER_QUEUES = (CYBERLAMP_CLUSTER_QUEUES
                        + ACI_CLUSTER_QUEUES
                        + OPENQ_CLUSTER_QUEUES)

ARRAY_RE = re.compile(r'(?P<body>.*)\.(?P<index>\d+)$')


def mkdir(d, mode=0o750):
    d = expand(d)
    try:
        makedirs(d, mode=mode)
    except OSError as err:
        if err.errno != 17:
            raise err


def get_xml_val(node, key):
    """Get string value in XML node by key, i.e. ``<key>val</key>``

    Note that this returns ONLY the first occurrence of `key`.

    Parameters
    ----------
    node : ElementTree node
    key : string

    Returns
    -------
    val : string

    """
    subnode = get_xml_subnode(node, key)
    if subnode is not None:
        val = subnode.text
    else:
        val = None
    return val


def get_xml_subnode(node, key):
    """Get sub-node in XML node by key, i.e. ``<node><subnode>...</subnode>``

    Note that this returns ONLY the first occurrence of `key`.

    Parameters
    ----------
    node : ElementTree node
    key : string

    Returns
    -------
    subnode : ElementTree node or None
        If the node could not be found, None is returned.

    """
    try:
        subnode = next(node.iter(key))
    except StopIteration:
        subnode = None
    return subnode


def convert_size(s):
    """Convert a qstat size string to int bytes.

    Parameters
    ----------
    s : string

    Returns
    -------
    size : int

    Notes
    -----
    See linux.die.net/man/7/pbs_resources for definition of size spec. Not
    implementing "word" logic here.

    """
    scales = dict(k=1024, m=1024**2, g=1024**3, t=1024**4, p=1024**5,
                  e=1024**6)
    num_re = re.compile(r'(?P<mag>\d+)(?P<scale>[kmgtpe]){0,1}(?:[b]{0,1})',
                        re.IGNORECASE)
    match = num_re.match(s)
    if match is None:
        raise ValueError('Failed to parse quantity "%s"' % s)
    groupdict = num_re.match(s).groupdict()
    if groupdict['scale'] is not None:
        factor = scales[groupdict['scale'].lower()]
    else:
        factor = 1
    return int(ceil(float(groupdict['mag']) * factor))


def get_qstat_output():
    """Get qstat output, caching by saving to a gzipped XML file such that
    calls within STALE_TIME will not call the actual `qstat` command again.

    Note that the cache file is written to
        QSTAT_CACHE_DIR/qstat.$USER.xml.gz

    Returns
    -------
    qstat_out : string

    """
    qstat_fname = 'qstat.%s.xml.gz' % USER
    qstat_fpath = join(QSTAT_CACHE_DIR, qstat_fname)
    qstat_out = None
    mkdir(QSTAT_CACHE_DIR)

    if isfile(qstat_fpath) and time() - getmtime(qstat_fpath) < STALE_TIME:
        try:
            with GzipFile(qstat_fpath, mode='r') as f:
                qstat_out = f.read()
        except Exception:
            pass
        else:
            return qstat_out

    qstat_out = check_output(['qstat', '-x'])
    with GzipFile(qstat_fpath, mode='w') as f:
        f.write(qstat_out)

    return qstat_out


def display_info(jobs):
    """Display info about jobs.

    Parameters
    ----------
    jobs : pandas.DataFrame

    """
    cluster_width = 12
    queue_width = 23
    number_width = 9
    field_widths = (
        -cluster_width, -queue_width, number_width, number_width, number_width
    )
    fmt = '  '.join('%' + str(s) + 's' for s in field_widths)

    print(fmt % ('Cluster', 'Queue Name', 'Running', 'Queued', 'Run+Queue'))
    print(fmt % tuple('-'*int(abs(s)) for s in field_widths))
    total_r = 0
    total_q = 0
    if 'cluster' not in jobs:
        print(fmt % ('Totals:', '', total_r, total_q, total_r + total_q))
        return

    for cluster, cgrp in jobs.groupby('cluster'):
        subtot_ser = cgrp.groupby('job_state')['job_state'].count()
        subtot = OrderedDict()
        subtot['R'] = subtot_ser.get('R', default=0)
        subtot['Q'] = subtot_ser.get('Q', default=0)
        total_r += subtot['R']
        total_q += subtot['Q']
        queue_num = 0
        for queue, qgrp in cgrp.groupby('queue'):
            if len(qgrp) == 0: # pylint: disable=len-as-condition
                continue
            queue_num += 1
            counts = qgrp.groupby('job_state')['job_state'].count()
            if queue_num == 1:
                cl = cluster
            else:
                cl = ''
            if len(queue) > queue_width:
                qn = queue[:queue_width]
            else:
                qn = queue

            q_counts = OrderedDict()
            q_counts['R'] = counts.get('R', default=0)
            q_counts['Q'] = counts.get('Q', default=0)

            print(fmt % (cl, qn, q_counts['R'], q_counts['Q'],
                         q_counts['R'] + q_counts['Q']))
        if queue_num > 1:
            print(fmt % ('', '> Subtotals:'.rjust(cluster_width), subtot['R'],
                         subtot['Q'], subtot['R']+subtot['Q']))
        print('')

    print(fmt % ('Totals:', '', total_r, total_q, total_r + total_q))


def get_jobs():
    """Get job info as a Pandas DataFrame. Loads the `jobs` DataFrame from disk
    if the cache file has been written less than `STALE_TIME` prior to call,
    otherwise this is regenerated and saved to disk for caching purposes.

    Returns
    -------
    jobs : pandas.DataFrame

    """
    pickle_name = 'qstat.%s.pkl' % USER
    pickle_path = join(QSTAT_CACHE_DIR, pickle_name)
    if isfile(pickle_path) and time() - getmtime(pickle_path) < STALE_TIME:
        jobs = pd.read_pickle(pickle_path)
        return jobs

    level1_keys = [
        'Job_Id', 'Job_Name', 'Job_Owner', 'job_state',
        'server', 'Account_Name', 'queue',
        'submit_args', 'submit_host', 'start_time', 'Walltime',
        'interactive', 'exit_status', 'exec_host', 'total_runtime',
        'init_work_dir'
    ]

    qstat_out = get_qstat_output()
    qstat_root = ElementTree.fromstring(qstat_out)

    jobs = []
    for job in qstat_root.iter('Job'):
        rec = OrderedDict()
        for key in level1_keys:
            low_key = key.lower()
            val = get_xml_val(job, key)
            rec[low_key] = val

        # Translate a couple of values to easier-to-use/understand values
        rec['job_owner'] = rec['job_owner'].split('@')[0]
        rec['full_job_id'] = rec['job_id']
        rec['job_id'] = int(rec['job_id'].split('.')[0])
        rec['walltime'] = pd.Timedelta(rec['walltime'])
        # TODO: make this a pd.Timestamp?
        #jobs['start_time'] = jobs['start_time'].astype('category')
        try:
            rec['total_runtime'] = pd.Timedelta(float(rec['total_runtime'])*1e9)
        except TypeError:
            #print(rec['total_runtime'], type(rec['total_runtime'])) # DEBUG
            rec['total_runtime'] = pd.Timedelta(np.nan)

        account_name = rec.pop('account_name')
        if account_name == 'cyberlamp':
            rec['cluster'] = 'cyberlamp'
            rec['queue'] = 'default'
        elif account_name in ACI_QUEUES:
            rec['cluster'] = 'aci'
            rec['queue'] = account_name.lower()
        else:
            raise ValueError('Unhandled account_name "%s"' % account_name)

        # Flatten hierarchical values: resources_used and resource_list

        resources_used = get_xml_subnode(job, 'resources_used')
        if resources_used is not None:
            for res in resources_used:
                res_name = res.tag.lower()
                res_val = res.text
                if 'mem' in res_name:
                    res_val = convert_size(res_val)
                elif 'time' in res_name or res_name == 'cput':
                    res_val = pd.Timedelta(res_val)
                elif res_name == 'energy_used':
                    continue
                rec['used_' + res_name] = res_val

        resource_list = get_xml_subnode(job, 'Resource_List')
        if resource_list is not None:
            for res in resource_list:
                res_name = res.tag.lower()
                res_val = res.text
                if 'mem' in res_name:
                    res_val = convert_size(res_val)
                elif 'time' in res_name or res_name in ['cput']:
                    res_val = pd.Timedelta(res_val) #*1e9)
                elif res_name == 'nodes':
                    fields = res_val.split(':')
                    rec['req_nodes'] = int(fields[0])
                    for field in fields[1:]:
                        name, val = field.split('=')
                        rec['req_' + name] = int(val)
                elif res_name == 'qos':
                    rec['qos'] = res_val
                rec['req_' + res_name] = res_val

            if rec['server'].endswith('aci.ics.psu.edu'):
                if rec['cluster'] == 'cyberlamp':
                    qos = get_xml_val(resource_list, 'qos')
                    if qos is None:
                        rec['queue'] = 'default'
                    else:
                        rec['queue'] = qos

        req_information = get_xml_subnode(job, 'req_information')
        if req_information is not None:
            contained_lists = OrderedDict()
            for req in req_information:
                req_name = req.tag.lower()
                if req_name.startswith('task_usage'):
                    continue
                match0 = ARRAY_RE.match(req_name)
                groupdict = match0.groupdict()
                #match1 = ARRAY_RE.match(groupdict['body'])
                req_name = groupdict['body'].replace('.', '_')
                req_val = req.text
                if req_name in ['task_count', 'lprocs']:
                    req_val = int(req_val)
                elif req_name == 'memory':
                    req_val = convert_size(req_val)
                if req_name not in contained_lists:
                    contained_lists[req_name] = []
                contained_lists[req_name].append(req_val)

            for req_name, lst in contained_lists.items():
                if len(lst) == 1:
                    rec[req_name] = lst[0]
                else:
                    rec[req_name] = ','.join(str(x) for x in lst)

        jobs.append(rec)

    jobs = pd.DataFrame(jobs)
    if len(jobs) == 0:
        return jobs

    jobs.sort_values([c for c in SORT_COLS if c in jobs.columns], inplace=True)

    # Manually convert dtypes of columns that auto convert can't figure out
    # (usually since first element might be `None` or `np.nan`
    if 'interactive' in jobs:
        jobs['interactive'] = jobs['interactive'].astype('category')
    if 'exit_status' in jobs:
        jobs['exit_status'] = jobs['exit_status'].astype('category')
    if 'qos' in jobs:
        jobs['qos'] = jobs['qos'].astype('category')
    if 'req_qos' in jobs:
        jobs['req_qos'] = jobs['req_qos'].astype('category')
    if 'exec_host' in jobs:
        jobs['exec_host'] = jobs['exec_host'].astype('category')

    # Auto-convert dtypes for the remaining columns
    convert_df_dtypes(jobs)

    jobs.to_pickle(pickle_path)

    return jobs


def get_openq_info():
    # Get OpenQ users
    openq_users = []
    with open(OPENQ_CONFIG_FPATH, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('list = '):
            openq_users = [s.strip() for s in line[7:].split(',')]
            break

    # Get number of files in users' OpenQ job dirs
    total_files = 0
    for user in openq_users:
        jobdir = join(HOME_ROOT, user, 'PBS', 'job_pool')
        try:
            user_files = len([f for f in listdir(jobdir) if isfile(f)])
        except (OSError, IOError):
            pass
        else:
            total_files += user_files

    # TODO: Find out which users are active openQ users
    cq = ('openq', 'openq')
    queues_status[cq]['queue_and_run_avail'] = 260*2 - total_files
    queues_status[cq]['run_avail'] = 0


if __name__ == '__main__':
    display_info(get_jobs())
