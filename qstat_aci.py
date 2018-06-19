#!/usr/bin/env python


"""
Parse qstat output and show pertinent info summary for ACI and CyberLAMP
clusters (e.g. on which sub-queues jobs are actually running).
"""


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from collections import Iterable, OrderedDict, Sequence
import errno
from fnmatch import fnmatch
from functools import partial
from getpass import getuser
from gzip import GzipFile
from math import ceil
from os import listdir, makedirs
from os.path import expanduser, expandvars, getmtime, isfile, join
import re
from subprocess import check_output
from time import time
from xml.etree import ElementTree

import numpy as np
import pandas as pd

#from pd_utils import convert_df_dtypes
#from pisa.utils.format import format_num

pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.options.display.max_colwidth = 9999
pd.options.display.width = 99999


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


__all__ = [
    'USER', 'SORT_COLS', 'QSTAT_CACHE_DIR', 'STALE_TIME', 'CYBERLAMP_QUEUES',
    'ACI_QUEUES', 'CYBERLAMP_CLUSTER_QUEUES', 'ACI_CLUSTER_QUEUES',
    'OPENQ_CLUSTER_QUEUES', 'KNOWN_CLUSTER_QUEUES', 'ARRAY_RE', 'STATE_TRANS',
    'DISPLAY_COLS', 'get_xml_val',
    'convert_size', 'get_qstat_output', 'get_jobs'
]


pd.set_option('display.max_rows', None)


def expand(p): # pylint: disable=invalid-name
    """Expand pathname"""
    return expanduser(expandvars(p))


def binpre_gi_formatter(x): # pylint: disable=invalid-name
    """Decimal-aligned Gi formatter including trailing zeros from "   0.001" Gi
    to "9999.999 Gi". """
    return '{:8.3f} Gi'.format(np.round(x / 1024**3, 3))


# yapf: disable
USER = getuser()
SORT_COLS = ['cluster', 'queue', 'job_owner', 'job_state', 'job_id']
QSTAT_CACHE_DIR = expand('/gpfs/group/dfc13/default/qstat_out')
STALE_TIME = 120 # seconds

CYBERLAMP_QUEUES = [
    'default',
    'cl_open',
    'cl_gpu',
    'cl_higpu',
    'cl_himem',
    'cl_debug',
    'cl_phi'
]
ACI_QUEUES = [
    'dfc13_a_g_sc_default',
    'dfc13_a_t_bc_default',
    'dfc13_b_g_sc_default',
    'dfc13_b_g_lc_default',
    'open',
]
OPENQ_QUEUES = [
    'openq'
]

CYBERLAMP_CLUSTER_QUEUES = [('cyberlamp', q) for q in CYBERLAMP_QUEUES]
ACI_CLUSTER_QUEUES = [('aci', q) for q in ACI_QUEUES]
OPENQ_CLUSTER_QUEUES = [(q, q) for q in OPENQ_QUEUES]
KNOWN_CLUSTER_QUEUES = (
    CYBERLAMP_CLUSTER_QUEUES
    + ACI_CLUSTER_QUEUES
    + OPENQ_CLUSTER_QUEUES
)

ARRAY_RE = re.compile(r'(?P<body>.*)\.(?P<index>\d+)$')

STATE_TRANS = {
    'queued': 'Q',
    'q': 'Q',
    'running': 'R',
    'r': 'R',
    's': 'S',
    'stopped': 'S',
    'c': 'C',
    'cancelled': 'C'
}

STATE_LABELS = OrderedDict([
    ('R', 'Running'),
    ('Q', 'Queued'),
    ('C', 'Completed'),
    ('S', 'Stopped'),
])

DISPLAY_COLS = [
    # General bits
    'cluster', 'queue', 'job_owner', 'job_state', 'job_id', 'job_name',
    # Host / processor
    'exec_host', 'lprocs',
    # Interactive job?
    'interactive',
    # Memory
    'req_mem', 'memory', 'used_mem', 'used_vmem',
    # Nodes, cpus, etc.
    'req_nodeset', 'req_nodect', 'req_ppn',
    # Time
    'start_time', 'req_walltime', 'used_walltime', 'used_cput',
    'total_runtime',
    # Accelerators, etc.
    'gpu_mode'
]
# yapf: enable


def mkdir(d, mode=0o750): # pylint: disable=invalid-name
    """Make directory and parents as necessary, setting permissions on all
    newly-created directories

    """
    d = expand(d)
    try:
        makedirs(d, mode=mode)
    except OSError as err:
        if err.errno != errno.EEXIST:
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


def convert_size(s): # pylint: disable=invalid-name
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


def get_qstat_output(force_refresh=False):
    """Get qstat output.

    Output is cached by saving to a gzipped XML file such that calls within
    STALE_TIME will not call the actual `qstat` command again, unless
    `force_refresh` is True..

    Note that the cache file is written to
        QSTAT_CACHE_DIR/qstat.$USER.xml.gz

    Parameters
    ----------
    force_refresh : bool
        Force retrieval of qstat info even if STALE_TIME has not passed since
        the last retrieval.

    Returns
    -------
    qstat_out : string

    """
    qstat_fname = 'qstat.%s.xml.gz' % USER
    qstat_fpath = join(QSTAT_CACHE_DIR, qstat_fname)
    qstat_out = None
    mkdir(QSTAT_CACHE_DIR)

    if (not force_refresh and isfile(qstat_fpath)
            and time() - getmtime(qstat_fpath) < STALE_TIME):
        try:
            with GzipFile(qstat_fpath, mode='r') as fobj:
                qstat_out = fobj.read()
        except Exception:
            pass
        else:
            return qstat_out

    qstat_out = check_output(['qstat', '-x'])
    with GzipFile(qstat_fpath, mode='w') as fobj:
        fobj.write(qstat_out)

    return qstat_out


def display_summary(jobs, states=None):
    """Display summary of job counts.

    Parameters
    ----------
    jobs : pandas.DataFrame

    states : string or iterable thereof, optional
        See keys and values of STATE_TRANS for valid state names. If not
        specified, defaults to display only "running" and "queued" jobs.
        Case insensitive.

    """
    if states is None:
        states = set(['R', 'Q', 'S'])
    else:
        if isinstance(states, basestring):
            states = [states]
        states = set(STATE_TRANS[s.strip().lower()] for s in states)

    n_states = len(states)
    if n_states == 1:
        totals_col = False
    else:
        totals_col = True

    cluster_width = 12
    queue_width = 23
    number_width = 9

    field_widths = [-cluster_width, -queue_width] + [number_width]*n_states
    if totals_col:
        field_widths.append(number_width)

    header = ['Cluster', 'Queue Name']
    totals_label = []
    ordered_states = []
    for state, label in STATE_LABELS.items():
        if state not in states:
            continue
        header.append(label)
        totals_label.append(state)
        ordered_states.append(state)
    if totals_col:
        header.append('+'.join(totals_label))

    fmt = '  '.join('%{}s'.format(fw) for fw in field_widths)

    print(fmt % tuple(header))
    print(fmt % tuple('-'*int(abs(s)) for s in field_widths))
    totals = OrderedDict()
    for state in ordered_states:
        totals[state] = 0
    if 'cluster' not in jobs:
        cols = ['Totals:', ''] + totals.values()
        if totals_col:
            cols.append(np.sum(totals.values()))
        print(fmt % tuple(cols))
        return

    for cluster, cgrp in jobs.groupby('cluster'):
        subtot_ser = cgrp.groupby('job_state')['job_state'].count()
        subtotals = OrderedDict()
        for state in ordered_states:
            state_ = subtot_ser.get(state, default=0)
            subtotals[state] = state_
            totals[state] += state_
        queue_num = 0
        for queue, qgrp in cgrp.groupby('queue'):
            if len(qgrp) == 0: # pylint: disable=len-as-condition
                continue
            queue_num += 1
            counts = qgrp.groupby('job_state')['job_state'].count()
            if queue_num == 1:
                cluster_ = cluster
            else:
                cluster_ = ''
            if len(queue) > queue_width:
                queue_ = queue[:queue_width]
            else:
                queue_ = queue

            q_counts = OrderedDict()
            for state in ordered_states:
                q_counts[state] = counts.get(state, default=0)

            cols = [cluster_, queue_] + q_counts.values()
            if totals_col:
                cols.append(np.sum(q_counts.values()))
            print(fmt % tuple(cols))

        if queue_num > 1:
            cols = (
                ['', '> Subtotals:'.rjust(cluster_width)]
                + subtotals.values()
            )
            if totals_col:
                cols.append(np.sum(subtotals.values()))
            print(fmt % tuple(cols))
        print('')

    cols = ['Totals:', ''] + totals.values()
    if totals_col:
        cols.append(np.sum(totals.values()))
    print(fmt % tuple(cols))


def get_jobs(users=None, cluster_queues=None, job_names=None, job_ids=None,
             states=None, force_refresh=False):
    """Get job info as a Pandas DataFrame.

    Loads the `jobs` DataFrame from disk if the cache file has been written
    less than `STALE_TIME` prior to call, otherwise this is regenerated and
    saved to disk for caching purposes.

    Returns
    -------
    jobs : pandas.DataFrame

    """
    pickle_name = 'qstat.%s.pkl' % USER
    pickle_path = join(QSTAT_CACHE_DIR, pickle_name)
    if (not force_refresh and isfile(pickle_path)
            and time() - getmtime(pickle_path) < STALE_TIME):
        jobs = pd.read_pickle(pickle_path)
        return jobs

    level1_keys = [
        'Job_Id', 'Job_Name', 'Job_Owner', 'job_state',
        'server', 'Account_Name', 'queue',
        'submit_args', 'submit_host', 'start_time', 'Walltime',
        'interactive', 'exit_status', 'exec_host', 'total_runtime',
        'init_work_dir'
    ]

    qstat_out = get_qstat_output(force_refresh=force_refresh)
    qstat_root = ElementTree.fromstring(qstat_out)

    jobs = []
    for job in qstat_root.iter('Job'):
        rec = OrderedDict()
        for key in level1_keys:
            low_key = key.lower()
            val = get_xml_val(job, key)
            rec[low_key] = val

        # Translate a couple of values to easier-to-use/understand values
        job_owner = rec['job_owner'].split('@')[0]
        if users is not None and job_owner not in users:
            continue
        rec['job_owner'] = job_owner
        rec['full_job_id'] = rec['job_id']
        rec['job_id'] = int(rec['job_id'].split('.')[0])
        rec['walltime'] = pd.Timedelta(rec['walltime'])
        # TODO: make this a pd.Timestamp?
        #jobs['start_time'] = jobs['start_time'].astype('category')
        try:
            rec['total_runtime'] = pd.Timedelta(float(rec['total_runtime'])*1e9)
        except TypeError:
            rec['total_runtime'] = pd.Timedelta(np.nan)

        account_name = rec.pop('account_name')
        if account_name == 'cyberlamp':
            rec['cluster'] = 'cyberlamp'
            rec['queue'] = 'default'
        elif account_name in ACI_QUEUES:
            rec['cluster'] = 'aci'
            rec['queue'] = account_name.lower()
        else:
            raise ValueError('Unhandled account_name "%s" owner "%s"'
                             % (account_name, rec['job_owner']))

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
                        split_str = field.split('=')
                        if len(split_str) == 1:
                            if 'gpu_mode' not in rec:
                                rec['gpu_mode'] = split_str[0]
                            else:
                                rec['gpu_mode'] += ', ' + split_str[0]
                        else:
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
    if len(jobs) == 0: # pylint: disable=len-as-condition
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

    start_time = None
    if 'start_time' in jobs:
        start_time = pd.to_datetime(jobs['start_time'], unit='s')
        del jobs['start_time']

    # Auto-convert dtypes for the remaining columns
    #convert_df_dtypes(jobs)

    if 'req_mem' in jobs:
        jobs['req_mem'] = jobs['req_mem'].astype('float')
    if 'memory' in jobs:
        jobs['memory'] = jobs['memory'].astype('float')
    if 'used_mem' in jobs:
        jobs['used_mem'] = jobs['used_mem'].astype('float')
    if 'used_vmem' in jobs:
        jobs['used_vmem'] = jobs['used_vmem'].astype('float')

    if start_time is not None:
        jobs['start_time'] = start_time

    jobs.to_pickle(pickle_path)

    return jobs


def query_jobs(jobs, users=None, cluster_queues=None, names=None, ids=None,
               states=None):
    """Run query on jobs dataframe.

    Parameters
    ----------
    jobs : pandas.DataFrame

    users : string or iterable of strings, optional
        Only retrieve jobs for the specified users; if None, retrieve for all
        users.

    cluster_queues : (str, str or None) or (str, (str, str, ...)); optional
        If None, return jobs from all queues on all clusters. Otherwise, only
        retrieve jobs for these (cluster, queue) or
        (cluster, (queue1, queue2, ...)) combinations. The queue can be None,
        in which case all queues from the specified cluster will be returned.

    names : string or iterable thereof, optional
        Only retrive these job names.

    ids : string or iterable thereof, optional
        Only retrive these job IDs.

    states : string in {'running', 'queued'} or iterable thereof, optional
        Only retrive jobs in these states.
        # Run queries where specified in function args

    Returns
    -------
    remaining_jobs : pandas.DataFrame

    """
    if isinstance(users, basestring):
        users = [users]

    if cluster_queues is not None:
        if (isinstance(cluster_queues, Sequence)
                and len(cluster_queues) == 2
                and isinstance(cluster_queues[0], basestring)):
            cluster_queues = [cluster_queues]

        cluster_queues_dict = dict()
        for cluster, queues in cluster_queues:
            assert isinstance(cluster, basestring)
            if isinstance(queues, basestring):
                queues = [queues]
            if isinstance(queues, Iterable):
                for queue in queues:
                    assert isinstance(queue, basestring)
                queues = set(queues)

            if cluster not in cluster_queues_dict:
                cluster_queues_dict[cluster] = queues
            elif queues is None or cluster_queues_dict[cluster] is None:
                cluster_queues_dict[cluster] = None
            else:
                cluster_queues_dict[cluster].update(queues)

        cluster_queues = cluster_queues_dict

    if isinstance(names, basestring):
        names = [names]

    if isinstance(ids, basestring):
        ids = [ids]

    if isinstance(states, basestring):
        states = [states]

    queries = []

    if users is not None:
        queries.append('job_owner in @users')

    if cluster_queues is not None:
        cq_queries = []
        for cluster, queues in cluster_queues.items():
            cq_query = cq_query = 'cluster == "{}"'.format(cluster)
            if queues is None:
                cq_queries.append(cq_query)
                continue
            for queue in queues:
                cq_queries.append('queue == "{}"'.format(queue))
            cq_query = '{} and ({})'.format(cq_query, ' | '.join(cq_queries))
            cq_queries.append(cq_query)
        queries.append(' | '.join('({})'.format(cq) for cq in cq_queries))

    if names is not None:
        queries.append('job_name in @names')

    if ids is not None:
        queries.append('job_id in @ids')

    if states is not None:
        queries.append('job_state in @states')

    if not queries or len(jobs) == 0:
        return jobs

    query = ' & '.join('({})'.format(q) for q in queries)
    remaining_jobs = jobs.query(query)
    return remaining_jobs


def get_openq_info():
    """Get info about jobs running via the OpenQ software"""
    # Get OpenQ users
    openq_users = []
    with open(OPENQ_CONFIG_FPATH, 'r') as fobj:
        lines = fobj.readlines()
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


def display_info(
        users=None, cluster_queues=None, names=None, ids=None,
        states=None, detail=None, sort=None, reverse=False,
        columns=False, force_refresh=False
    ):
    """Retrieve and display info about jobs.

    Parameters
    ----------
    users : string, iterable thereof, or None

    cluster_queues : string, iterable thereof, or None

    names :  : string, iterable thereof, or None

    ids :  : string, iterable thereof, or None

    states : string, iterable thereof, or None

    detail : None, bool, empty iterable, string, or iterable of strings
        If None or False, display is just a summary of total jobs in each
        state. If True or empty iterable, display detail using `DISPLAY_COLS`.
        If string or iterable of strings, display detail using those specified
        columns.

    sort : string, iterable thereof, or None
        If None, sorting defaults to the order in which columns are displayed.
        If string(s) are specified for `sort`, then sorting is in the order of
        the string(s) specified and then continues in the order of the
        remaining displayed columns. Note that you can specify sort columns
        that are not to be displayed.

    reverse : bool
        Sort in reverse (i.e., sort in descending order).

    columns : bool
        Display only the valid column names and return.

    force_refresh : bool
        Force reloading qstat info even if the last refresh was less than
        `STALE_TIME` ago.

    """
    if isinstance(states, basestring):
        states = [states]
    if isinstance(states, Iterable):
        states = set(STATE_TRANS[s.strip().lower()] for s in states)
    if users is not None and isinstance(users, basestring):
        users = [users]

    jobs = get_jobs(force_refresh=force_refresh, users=users)

    all_columns = set(str(c) for c in jobs.columns.values)
    all_columns_str = ' '.join(sorted(all_columns))

    if columns:
        print(all_columns_str)
        return

    remaining_jobs = query_jobs(
        jobs, users=users, cluster_queues=cluster_queues, names=names, ids=ids,
        states=states
    )
    remaining_jobs.index += 1

    #float_format = partial(
    #    format_num,
    #    sigfigs=4,
    #    fmt='binpre',
    #    sci_thresh=(3, -3),
    #    trailing_zeros=False,
    #    nanstr='--'
    #)

    if detail is None or detail == False:
        display_summary(remaining_jobs, states=states)
        return

    if detail == True:
        detail = []
    else:
        detail = list(detail)
    if len(detail) == 0:
        display_cols = [c for c in DISPLAY_COLS if c in all_columns]
    else:
        invalid = set(detail).difference(all_columns)
        if invalid:
            invalid = ', '.join(c for c in invalid)
            raise ValueError(
                'Invalid `detail` column(s) specified: {}'.format(invalid)
            )
        display_cols = detail

    # total_runtime only applies to completed jobs
    if states is not None and 'C' not in states and 'total_runtime' in display_cols:
        display_cols.remove('total_runtime')

    if not sort: # captures sort=[] or sort=None
        sort_cols = display_cols
    else:
        invalid = set(sort).difference(all_columns)
        if invalid:
            invalid = ', '.join(c for c in invalid)
            raise ValueError(
                'Invalid `sort` column(s) specified: {}'.format(invalid)
            )
        sort_cols = sort + [c for c in display_cols if c not in sort]

    if len(remaining_jobs) == 0:
        return

    remaining_jobs = (
        remaining_jobs
        .sort_values(by=sort_cols, ascending=not reverse)[display_cols]
        .reset_index(drop=True)
    )
    remaining_jobs.index += 1

    if len(display_cols) == 1:
        print(remaining_jobs[detail].to_string(header=False, index=False,
                                               na_rep='--'))
        return

    print(remaining_jobs.to_string(na_rep='--', float_format=binpre_gi_formatter))


def parse_args(description=__doc__):
    """Parse command line options"""
    argument_parser = ArgumentParser(description=description)
    argument_parser.add_argument(
        '--user', nargs='+',
        help='''Only retrieve jobs for username(s) specified'''
    )
    argument_parser.add_argument(
        '--cq', nargs='+', action='append',
        help='''Either specify just
            --cq cluster_name
        to select all queues on a cluster or specify
            --cq cluster_name q1_name q2_name ...
        to specify certain queues on a cluster. Repeat the --cq option multiple
        times to specify multiple clusters.'''
    )
    argument_parser.add_argument(
        '--name', nargs='+',
        help='''Job name(s) to return. Glob expansion is performed on the
        passed strings (specify inside of single quotes to avoid shell
        expansion of special glob characters, like '*')'''
    )
    argument_parser.add_argument(
        '--id', nargs='+',
        help='''Job ID(s) to return. Glob expansion is performed on the
        passed strings (specify inside of single quotes to avoid shell
        expansion of special glob characters, like '*')'''
    )
    argument_parser.add_argument(
        '--state', choices=sorted(STATE_TRANS.keys()), nargs='+',
        help='''Job state(s) to return. Valid choices are "running" and
        "queued". If not specified, all are returned.'''
    )
    argument_parser.add_argument(
        '--detail', nargs='*', default=None,
        help='''Print full detail. If not specified, a summary is printed.
        Provide arguments to --detail to specify the column(s) to display.
        Values are sorted according to the order of columns specified.'''
    )
    argument_parser.add_argument(
        '--sort', nargs='+', default=None,
        help='''Sort by these columns first; sort continues on remaining
        columns specified by --detail, in order they are displayed.'''
    )
    argument_parser.add_argument(
        '-r', '--reverse', action='store_true',
        help='''Reverse the sort (i.e., sort in descending order).'''
    )
    argument_parser.add_argument(
        '--columns', action='store_true',
        help='''Display all column names possible to specify.'''
    )
    argument_parser.add_argument(
        '-f', '--force', action='store_true',
        help='''Force refresh of qstat information, even if {} sec have not
        passed since the last refresh.'''.format(STALE_TIME)
    )
    args = argument_parser.parse_args()
    return args


def main():
    """Run display_info as a script"""
    args = parse_args()
    args_d = vars(args)
    cqs = args_d.pop('cq')
    if cqs is None:
        cluster_queues = None
    else:
        cluster_queues = []
        for cq in cqs:
            if len(cq) == 1:
                cluster_queues.append((cq[0], None))
            else:
                cluster_queues.append((cq[0], cq[1:]))
    args_d['cluster_queues'] = cluster_queues
    args_d['users'] = args_d.pop('user')
    args_d['ids'] = args_d.pop('id')
    args_d['names'] = args_d.pop('name')
    args_d['states'] = args_d.pop('state')
    args_d['force_refresh'] = args_d.pop('force')
    display_info(**args_d)


if __name__ == '__main__':
    main()
