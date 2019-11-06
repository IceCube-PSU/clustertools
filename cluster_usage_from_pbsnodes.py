#!/usr/bin/env python
# pylint: disable=redefined-outer-name, wrong-import-position

"""
Retrieve and report cluster status, including generation of online plots.
"""


from __future__ import absolute_import, division, print_function


__author__ = 'J.L. Lanfranchi'
__license__ = '''Copyright (c) 2018, Justin L. Lanfranchi

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


from bz2 import BZ2File
from collections import OrderedDict
from datetime import datetime as dtmod
from itertools import product
import os
from os.path import basename, getctime, isdir, isfile, join
import re
from socket import gethostname
from subprocess import Popen, PIPE
import time
from xml.etree import ElementTree

from six import string_types

import numpy as np
import pandas as pd

try:
    from plotly import graph_objs, plotly
except ImportError:
    from chart_studio import plotly
    from plotly import graph_objects as graph_objs

from qstat_aci import expand, mkdir, get_xml_subnode, convert_size


__all__ = [
    'DEBUG',
    'STALE_SEC',
    'HOSTNAME',
    'ON_ACI',
    'REMOTE_HOST',
    'CACHE_DIR',
    'GPU_RE',
    'CL_TOTAL_GPUS',
    'CL_NAME_PFX',
    'CLUSTER_SUBGROUP_PPTY_MAPPING',
    'PPTY_CLUSTER_MAPPING',
    'make_datetime_stamp',
    'pbsnodes2dataframes',
    'cache_node_info',
    'summarize_gpu_info',
    'compute_open_slots',
    'get_pbsnodes_output',
    'get_cached_info',
    'get_info',
    'plot_slot_avail',
]


DEBUG = True

STALE_SEC = 300
HOSTNAME = gethostname()
ON_ACI = HOSTNAME.endswith('acib.ics.psu.edu') or HOSTNAME.endswith('aci.ics.psu.edu')
REMOTE_HOST = None if ON_ACI else 'aci-b.aci.ics.psu.edu'
CACHE_DIR = (
    '/gpfs/group/dfc13/default/pbsnodes_cache' if ON_ACI
    else expand('~/.cache/pbsnodes')
)
GPU_RE = re.compile(r'gpu\[([0-9]+)\]=')
CL_TOTAL_GPUS = 101
CL_NAME_PFX = 'comp-clgc-'
CLUSTER_SUBGROUP_PPTY_MAPPING = dict(
    aci=dict(
        legacy='legacy',
        basic='basic',
        stmem='stmem', # includes 'sthaswell' and 'stivybridge'
        himem='himem',
    ),
    cyberlamp=dict(
        basic='clbasic',
        higpu='clhigpu',
        himem='clhimem',
        phi='clphi',
    ),
)
PPTY_CLUSTER_MAPPING = {}
for _cl, _sg in CLUSTER_SUBGROUP_PPTY_MAPPING.items():
    for _sg_name, _ppty in _sg.items():
        PPTY_CLUSTER_MAPPING[_ppty] = (_cl, _sg_name)


def make_datetime_stamp(datetime=None, human_readable=False):
    """Generate ISO-8601 date-time stamp.

    If `datetime` is provided, the date-time stamp is generated for the
    specified time.

    Parameters
    ----------
    datetime : int or string, optional
        Seconds since the epoch (int) or datetime (string).

    human_readable : bool
        Make a human-readable date-time stamp (as opposed to a date-time stamp
        meant to be used in a filename).

    Returns
    -------
    datetime_stamp : string
    sec_since_epoch : int

    """
    # NOTE: We do the "dumb" thing here and assume all times are in local time,
    #       so we simply strip any timezone and attach the local timezone.

    if isinstance(datetime, string_types):
        tz_re = re.compile(r'([+-]([0-9]{4}))|([ ][a-z]{3})$', re.IGNORECASE)
        datetime = tz_re.sub('', datetime)
        try:
            time_struct = dtmod.strptime(datetime, '%Y-%m-%dT%H%M%S')
        except ValueError:
            time_struct = dtmod.strptime(datetime, '%Y-%m-%d %H:%M:%S')
        time_struct = time_struct.timetuple()
    elif isinstance(datetime, (int, float)):
        time_struct = time.localtime(datetime)
    else:
        time_struct = time.localtime()

    sec_since_epoch = time.mktime(time_struct)

    if human_readable:
        datetime_stamp = '{} {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time_struct),
            time.strftime('%Z')
        )
    else:
        datetime_stamp = '{}{}'.format(
            time.strftime('%Y-%m-%dT%H%M%S', time_struct),
            time.strftime('%z')
        )

    return datetime_stamp, sec_since_epoch


def pbsnodes2dataframes(pbsnodes_xml):
    """Convert xml output of pbsnodes to a Pandas DataFrame.

    Parameters
    ----------
    pbsnodes_xml : string
        Either interpret as a filename and open an existing output or interpret
        as an XML string (whichever works).

    Returns
    -------
    node_info : pandas.DataFrame
    sec_since_epoch : None or float

    """
    sec_since_epoch = None
    if DEBUG:
        t0 = time.time()
        print('>> pbsnodes2dataframes')

    if isfile(pbsnodes_xml):
        if DEBUG:
            print('>>   reading pbsnodes_xml from "{}"'.format(pbsnodes_xml))
        pbsnodes_xml, sec_since_epoch = get_pbsnodes_output(
            fpath=pbsnodes_xml, stale_sec=np.inf
        )
        if DEBUG:
            print('>>   sec_since_epoch =', sec_since_epoch)

    root = ElementTree.fromstring(pbsnodes_xml)
    data = get_xml_subnode(root, 'Data')

    node_info = []
    gpu_info = []
    for node in data.iter('Node'):
        this_node_info = OrderedDict()

        for element in node.iter():
            this_node_info[element.tag] = element.text

        if 'status' in this_node_info:
            status = this_node_info.pop('status')
            fields = status.split(',')
            for field in fields:
                if field.startswith('message') or field.startswith('note'):
                    continue
                try:
                    key, val = field.split('=')
                except:
                    print('parsing field failed:\n{}\n'.format(field))
                    continue
                if key == 'jobs':
                    continue
                this_node_info[key] = val

        if 'properties' in this_node_info:
            this_node_info['properties'] = tuple(
                sorted(this_node_info['properties'].split(','))
            )

        if 'jobs' in this_node_info:
            allocated_cores = 0
            jobs = this_node_info.pop('jobs').split(',')
            for job in jobs:
                cores = job.split('/')[0]
                try:
                    c0 = int(cores)
                    cores = 1
                except ValueError:
                    c0, c1 = cores.split('-')
                    cores = int(c1) - int(c0) + 1
                allocated_cores += cores
            this_node_info['allocated_cores'] = allocated_cores

        if 'gpu_status' in this_node_info:
            gpu_status = this_node_info.pop('gpu_status')
            ginfos = GPU_RE.split(gpu_status)[1:]

            # Keep track of high-level GPU stats for just this node
            n_gpus = 0
            gpus_unalloc = 0
            gpus_alloc = 0
            gpu_total_util = 0
            gpu_alloc_util = 0

            for _, ginfo in zip(ginfos[::2], ginfos[1::2]):
                this_gpu_info = OrderedDict()
                this_gpu_info['name'] = this_node_info['name']
                this_gpu_info['properties'] = this_node_info['properties']
                useful_bits = ginfo.split(',')[0].split(';')
                for bits in useful_bits:
                    key, val = bits.split('=')
                    if key in ['gpu_memory_total', 'gpu_memory_used']:
                        val = convert_size(val)
                    elif 'utilization' in key:
                        val = int(val.rstrip('%'))
                    elif 'ecc_errors' in key:
                        val = int(val)
                    elif key == 'gpu_temperature':
                        val = int(val.rstrip(' C'))
                    this_gpu_info[key] = val
                gpu_info.append(this_gpu_info)

                if this_gpu_info['gpu_state'] == 'Unallocated':
                    gpus_unalloc += 1
                else:
                    gpu_alloc_util += float(this_gpu_info['gpu_utilization'])
                    gpus_alloc += 1
                gpu_total_util += float(this_gpu_info['gpu_utilization'])

            n_gpus = gpus_alloc + gpus_unalloc
            this_node_info['gpus_alloc'] = gpus_alloc
            this_node_info['gpus_unalloc'] = gpus_unalloc
            this_node_info['gpu_alloc_util'] = (
                gpu_alloc_util / gpus_alloc if gpus_alloc else 0.0
            )
            this_node_info['gpu_total_util'] = gpu_total_util / n_gpus

        # The "note" and "message" fields don't abide by other feilds'
        # formatting conventions (arbitrary user text), and we don't use them
        # anway, so drop it before further parsing
        if 'note' in this_node_info:
            this_node_info.pop('note')
        if 'message' in this_node_info:
            this_node_info.pop('message')

        for key, val in this_node_info.items():
            if isinstance(val, string_types) and val.endswith('kb'):
                this_node_info[key] = int(convert_size(val) / 1024**3)

        if 'properties' not in this_node_info:
            print('Node has no "properties" key, skipping:', this_node_info)
            continue

        # Assign a cluster name and subgroup name to the node
        cluster, subgroup = None, None
        for ppty in this_node_info['properties']:
            if ppty not in PPTY_CLUSTER_MAPPING:
                continue
            cluster, subgroup = PPTY_CLUSTER_MAPPING[ppty]
            break

        if cluster is not None:
            this_node_info['cluster'] = cluster
            this_node_info['subgroup'] = subgroup

        node_info.append(this_node_info)

    node_info = pd.DataFrame(node_info)
    node_info = node_info[~node_info.opsys.isnull()]

    fill_na_with_zero_fields = [
        'gpus', 'allocated_cores', 'gpus_alloc', 'gpus_unalloc'
    ]
    int_fields = [
        'np', 'total_sockets', 'total_numa_nodes', 'total_cores',
        'total_threads', 'dedicated_sockets', 'dedicated_numa_nodes',
        'dedicated_cores', 'dedicated_threads', 'nsessions', 'nusers',
        'idletime', 'ncpus', 'netload', 'rectime', 'gpus', 'gpus_alloc',
        'gpus_unalloc', 'allocated_cores', 'totmem', 'availmem', 'physmem'
    ]
    float_fields = [
        'loadave', 'gpu_alloc_util', 'gpu_total_util'
    ]
    drop_fields = [
        'Node', 'message', 'macaddr', 'ntype', 'mom_service_port',
        'mom_manager_port', 'uname', 'gres', 'varattr', 'np', 'ncpus',
        'opsys'
    ]
    categ_fields = [
        'state', 'power_state', 'properties'
    ]

    for col in fill_na_with_zero_fields:
        node_info[col].fillna(0, inplace=True)

    for col in int_fields:
        try:
            node_info[col] = node_info[col].astype(int)
        except ValueError:
            print('Column "%s" could not be converted to int' % col)

    for col in float_fields:
        node_info[col] = node_info[col].astype(float)

    for col in categ_fields:
        node_info[col] = node_info[col].astype('category')

    node_info = node_info[[c for c in node_info.columns
                           if c not in drop_fields]]

    node_info['cores_unalloc'] = (
        node_info.total_cores - node_info.allocated_cores
    )

    # Rename columns
    col_name_mapping = dict(
        total_sockets='sockets',
        total_numa_nodes='numa_nodes',
        total_cores='cores',
        availmem='mem_avail',
        physmem='mem_phys',
        totmem='mem_tot',
        allocated_cores='cores_alloc'
    )
    node_info.rename(index=str, columns=col_name_mapping, inplace=True)

    # Reorder columns
    col_order = [
        'cluster', 'subgroup', 'name', 'properties',
        'state', 'power_state',
        'cores', 'cores_alloc', 'cores_unalloc',
        'gpus', 'gpus_alloc', 'gpus_unalloc',
    ]
    col_order = (
        [c for c in col_order]
        + sorted([c for c in node_info.columns if c not in col_order])
    )
    node_info = node_info[col_order]

    # Postprocessing to cleanup `gpu_info` DataFrame
    gpu_info = pd.DataFrame(gpu_info)

    int_fields = [
        'gpu_pci_device_id'
    ]
    categ_fields = [
        'gpu_pci_location_id', 'gpu_product_name', 'gpu_mode', 'gpu_state',
        'gpu_ecc_mode'
    ]
    drop_fields = [
        'gpu_id', 'gpu_display'
    ]

    for col in int_fields:
        try:
            gpu_info[col] = gpu_info[col].astype(int)
        except ValueError:
            print('Column "%s" could not be converted to int' % col)

    for col in categ_fields:
        gpu_info[col] = gpu_info[col].astype('category')

    gpu_info = gpu_info[[c for c in gpu_info.columns if c not in drop_fields]]

    if DEBUG:
        print('>>   pbsnodes2dataframes:', time.time() - t0)

    return node_info, gpu_info, sec_since_epoch


def cache_node_info(node_info, gpu_info, sec_since_epoch=None,
                    cache_dir=CACHE_DIR):
    """Cache `node_info` to `cache_dir` with filename constructed to include
    current date & time.

    Parameters
    ----------
    node_info : pandas.DataFrame

    gpu_info : pandas.DataFrame

    sec_since_epoch : scalar or string, optional
        If provided, use this for date-time stamp to place on the cache files.
        If None, a new date-time stamp is generated from the current time.

    cache_dir : string, optional

    Returns
    -------
    sec_since_epoch

    """
    if DEBUG:
        print('>> cache_node_info')
    cache_dir = expand(cache_dir)
    mkdir(cache_dir)

    datetime_stamp, sec_since_epoch = make_datetime_stamp(sec_since_epoch)
    for pfx, df in zip(['node', 'gpu'], [node_info, gpu_info]):
        fbasename = '{}.{}_info.pkl.bz2'.format(datetime_stamp, pfx)
        fpath = join(cache_dir, fbasename)
        if DEBUG:
            print('>>   Caching {}_info to "{}"'.format(pfx, fpath))
        df.to_pickle(
            fpath, compression='bz2'
        )

    return sec_since_epoch


def summarize_gpu_info(gpu_info):
    """CyberLAMP GPU summary.

    Parameters
    ----------
    gpu_info : pandas.DataFrame

    Returns
    -------
    summary : OrderedDict

    """
    summary = OrderedDict()

    gpu_info = gpu_info[gpu_info.name.str.contains(CL_NAME_PFX)]
    counts = gpu_info.groupby('gpu_state').count()['name']
    dead = int(CL_TOTAL_GPUS - counts.sum())
    counts = counts.to_dict()
    counts['Dead'] = dead

    for key in sorted(counts.keys()):
        summary[key] = counts[key]

    unallocated = gpu_info.query('gpu_state == "Unallocated"')
    allocated = gpu_info.query('gpu_state != "Unallocated"')

    summary['total_util'] = gpu_info.gpu_utilization.sum() / CL_TOTAL_GPUS
    summary['alloc_util'] = allocated.gpu_utilization.mean()

    summary['mem_total_util'] = (
        gpu_info.gpu_memory_utilization.sum() / CL_TOTAL_GPUS
    )
    summary['mem_alloc_util'] = allocated.gpu_memory_utilization.mean()

    summary['mean_temp'] = gpu_info.gpu_temperature.mean()
    summary['mean_temp_unalloc'] = unallocated.gpu_temperature.mean()
    summary['mean_temp_alloc'] = allocated.gpu_temperature.mean()

    return summary


def compute_open_slots(node_info, slot_cores, slot_mem, slot_gpus=0):
    """Compute how many slots across the cluster are available (or at least
    should be available) given the requested resources.

    Parameters
    ----------
    node_info : pandas.DataFrame
    slot_cores
    slot_mem
        Value(s) in units of GiB
    slot_gpus : scalar int >= 0
        Must be a single value

    """
    assert np.isscalar(slot_gpus)

    slots_avail = np.zeros((slot_cores.size, slot_mem.size), dtype=int)
    for _, node in node_info.iterrows():
        mem_slots = node.mem_avail // slot_mem
        core_slots = node.cores_unalloc // slot_cores
        this_slots_avail = np.minimum.outer(core_slots, mem_slots)
        if slot_gpus > 0:
            gpu_slots = node.gpus_unalloc // slot_gpus
            this_slots_avail = np.minimum(gpu_slots, this_slots_avail)
        slots_avail += this_slots_avail
    return slots_avail


def get_pbsnodes_output(fpath=None, remote_host=None, stale_sec=STALE_SEC,
                        cache_dir=CACHE_DIR):
    """Retrieve output of pbsnodes if current time is `stale_sec` later than
    any cached pbsnodes output; otherwise, load from the latest cache file.

    Will check via ssh on a remote host if `remote_host` is specified.

    Parameters
    ----------
    fpath : string, optional
        If provided, output is loaded from the file (and stale_sec is ignored).
        The `sec_since_epoch` is attempted to be obtained from the filename or
        the file's creation time.

    remote_host : string, optional
        Whether to run `pbsnodes` command on a remote host.

    stale_sec : scalar, optional
        Seconds to mark a cache file as stale and therefore needing to reload
        the output of the pbsnodes command. Force reloading by setting
        `stale_sec=0` and force loading from a cache file (if one exists) by
        setting `stale_sec=np.inf`.

    cache_dir : string, optional
        Directory from which / into which to load / store cache files.

    Returns
    -------
    pbsnodes_xml : string
        Output of `pbsnodes -x`

    sec_since_epoch : None or float
        Time at which the pbsnodes command was run that produced the returned
        `pbsnodes_xml`. Returns None if this info could not be ascertained.

    """
    t0 = time.time()
    if DEBUG:
        print('>> get_pbsnodes_output')

    cache_dir = expand(cache_dir)

    sec_since_epoch = None
    if fpath is not None:
        if DEBUG:
            print('>>   loading pbsnodes xml from', fpath)
        try:
            datetime_stamp = basename(fpath).split('.')[0]
            _, sec_since_epoch = make_datetime_stamp(datetime_stamp)
        except Exception:
            # File creation time
            sec_since_epoch = getctime(fpath)
        pbsnodes_xml = open(fpath, 'r').read()
        return pbsnodes_xml, sec_since_epoch

    pbsnodes_files = []
    if isdir(cache_dir):
        contents = os.listdir(cache_dir)
        pbsnodes_files = sorted(
            [f for f in contents if f.endswith('.pbsnodes.xml.bz2')]
        )

    if pbsnodes_files:
        cached_pbsnodes_f = pbsnodes_files[-1]
        if DEBUG:
            print('>>   cached_pbsnodes_f:', cached_pbsnodes_f)

        try:
            datetime_stamp = basename(cached_pbsnodes_f).split('.')[0]
            _, sec_since_epoch = make_datetime_stamp(datetime_stamp)
        except:
            sec_since_epoch = getctime(cached_pbsnodes_f)

        offset = t0 - sec_since_epoch
        if offset <= stale_sec:
            if DEBUG:
                print('>>   reading from cached_pbsnodes_f')
            fpath = join(cache_dir, cached_pbsnodes_f)
            with BZ2File(fpath, 'r') as bz2_file:
                pbsnodes_xml = bz2_file.read()
            return pbsnodes_xml, sec_since_epoch
        if DEBUG:
            print('>>   cached_pbsnodes_f is stale; getting pbsnodes output')

    datetime_stamp, sec_since_epoch = make_datetime_stamp(t0)

    if remote_host:
        command = ['ssh', remote_host, 'pbsnodes -x']
    else:
        command = ['pbsnodes', '-x']
    proc = Popen(command, stdout=PIPE, stderr=PIPE)
    pbsnodes_xml, stderr = proc.communicate()
    if proc.returncode != 0:
        raise Exception(stderr)

    mkdir(cache_dir)
    fpath = join(cache_dir, '{}.pbsnodes.xml.bz2'.format(datetime_stamp))
    if DEBUG:
        print('>>   sec_since_epoch', sec_since_epoch)
        print('>>   Caching pbsnodes XML output to "{}"'.format(fpath))
    with BZ2File(fpath, 'w') as bz2_file:
        bz2_file.write(pbsnodes_xml)

    return pbsnodes_xml, sec_since_epoch


def get_cached_info(stale_sec=STALE_SEC, cache_dir=CACHE_DIR):
    """Get node_info and gpu_info DataFrames stored to disk (if their files
    exist and they are not stale).

    Parameters
    ----------
    stale_sec : scalar
        Files older than this many seconds ago are considered stale.

    cache_dir : string, optional

    Returns
    -------
    node_info, gpu_info : pandas.DataFrame
    sec_since_epoch : float
        Time at which the node_info file was generated.

    Raises
    ------
    ValueError
        If either node_info or gpu_info cache files are stale.

    """
    t0 = time.time()
    if DEBUG:
        print('>> get_cached_info')

    cache_dir = expand(cache_dir)
    if not isdir(cache_dir):
        raise ValueError('Cache directory does not exist, so no cached files!')

    contents = os.listdir(cache_dir)
    node_info_files = sorted(
        [f for f in contents if f.endswith('.node_info.pkl.bz2')]
    )
    gpu_info_files = sorted(
        [f for f in contents if f.endswith('.gpu_info.pkl.bz2')]
    )

    if not (node_info_files or gpu_info_files):
        raise ValueError()

    cached_gpu_info_f = gpu_info_files[-1]
    if DEBUG:
        print('>>   cached_gpu_info_f:', cached_gpu_info_f)
    datetime_stamp = basename(cached_gpu_info_f).split('.')[0]
    _, sec_since_epoch = make_datetime_stamp(datetime_stamp)
    offset = t0 - sec_since_epoch
    if offset > stale_sec:
        raise ValueError('Cached gpu info is stale')

    cached_node_info_f = node_info_files[-1]
    if DEBUG:
        print('>>   cached_node_info_f:', cached_node_info_f)
    datetime_stamp = basename(cached_node_info_f).split('.')[0]
    _, sec_since_epoch = make_datetime_stamp(datetime_stamp)
    offset = t0 - sec_since_epoch
    if DEBUG:
        print('>>   offset:', offset)
    if offset > stale_sec:
        raise ValueError('Cached node info is stale')

    if DEBUG:
        print('>>   loading from cached_{gpu,node}_info_f')

    node_info = pd.read_pickle(join(cache_dir, cached_node_info_f))
    gpu_info = pd.read_pickle(join(cache_dir, cached_gpu_info_f))

    if DEBUG:
        print('>>   get_cached_info:', time.time() - t0)

    return node_info, gpu_info, sec_since_epoch


def get_info(pbsnodes_xml=None, sec_since_epoch=None, node_info_fpath=None,
             gpu_info_fpath=None, stale_sec=STALE_SEC, cache_dir=CACHE_DIR,
             remote_host=REMOTE_HOST):
    """
    Parameters
    ----------
    pbsnodes_xml : string, optional
        XML output of pbsnodes command or filepath to pbsnodes cache file. If
        specified, neither `node_info_fpath` nor `gpu_info_fpath` can be
        specified.

    sec_since_epoch : scalar, optional
        Timestamp for pbsnodes_xml if `pbsnodes_xml` is the raw XML string or
        or if the timestamp cannot be ascertained from `pbsnodes_xml`.
        Otherwise, the timestamp will be set to current time.

    node_info_fpath, gpu_info_fpath : string, optional
        Path to cache files containing node_info and gpu_info DataFrames. If
        not specified, the respective DataFrame is regenerated from pbsnodes
        XML.

    stale_sec : scalar, optional

    cache_dir : string, optional

    remote_host : string, optional


    Returns
    --------
    node_info, gpu_info : pandas.DataFrame
    sec_since_epoch : float or None

    """
    t0 = time.time()
    node_info, gpu_info = None, None
    if DEBUG:
        print('>> get_info')

    if pbsnodes_xml is not None:
        assert node_info_fpath is None
        assert gpu_info_fpath is None
        node_info, gpu_info, sec_since_epoch_ = pbsnodes2dataframes(
            pbsnodes_xml=pbsnodes_xml
        )
        if sec_since_epoch_ is not None:
            sec_since_epoch = sec_since_epoch_

    if gpu_info_fpath is not None:
        if DEBUG:
            print('>>   loading gpu_info df from', node_info_fpath)
        try:
            datetime_stamp = basename(gpu_info_fpath).split('.')[0]
            _, sec_since_epoch = make_datetime_stamp(datetime_stamp)
        except Exception:
            sec_since_epoch = getctime(gpu_info_fpath)
        gpu_info = pd.read_pickle(gpu_info_fpath)

    if node_info_fpath is not None:
        if DEBUG:
            print('>>   loading node_info df from', node_info_fpath)
        try:
            datetime_stamp = basename(node_info_fpath).split('.')[0]
            _, sec_since_epoch = make_datetime_stamp(datetime_stamp)
        except Exception:
            sec_since_epoch = getctime(node_info_fpath)
        node_info = pd.read_pickle(node_info_fpath)

    if node_info is None or gpu_info is None:
        if DEBUG:
            print('>>   getting cached node/gpu info')
        try:
            node_info_, gpu_info_, sec_since_epoch_ = get_cached_info(
                stale_sec=STALE_SEC, cache_dir=cache_dir
            )
        except ValueError:
            if DEBUG:
                print(
                    '>>   failed to get cached file (stale or missing);'
                    ' regnerating'
                )
            pbsnodes_xml, sec_since_epoch_ = get_pbsnodes_output(
                remote_host=remote_host,
                stale_sec=stale_sec,
                cache_dir=cache_dir
            )
            node_info_, gpu_info_, _ = pbsnodes2dataframes(pbsnodes_xml)
            cache_node_info(
                node_info=node_info_,
                gpu_info=gpu_info_,
                sec_since_epoch=sec_since_epoch_
            )

        if node_info is None:
            node_info = node_info_
            sec_since_epoch = sec_since_epoch_

        if gpu_info is None:
            gpu_info = gpu_info_

    if DEBUG:
        print('>>   get_info', time.time() - t0)

    return node_info, gpu_info, sec_since_epoch


def plot_slot_avail(node_info, sec_since_epoch=None):
    """Plot "slot" availability.

    Parameters
    ----------
    node_info : pandas.DataFrame
    sec_since_epoch : numeric, optional
        If provided, a date-time stamp will be displayed on the plots
        indicating the time the information displayed was generated.

    """
    if DEBUG:
        t0 = time.time()
        print('>> plot_slot_avail')

    updated_at = ''
    if sec_since_epoch is not None:
        datetime_stamp, _ = make_datetime_stamp(
            sec_since_epoch, human_readable=True
        )
        updated_at = 'Updated {}'.format(datetime_stamp)

    slot_cores = np.array(
        [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40]
    )
    slot_mem = 2**np.arange(0, 11)

    cluster_node_info = OrderedDict([
        ('ACI CPU', node_info.query('cluster == "aci"')),
        ('CyberLAMP CPU-Only',
         node_info.query('(cluster == "cyberlamp") & (subgroup != "phi")')),
        ('CyberLAMP Single-GPU',
         node_info.query('(cluster == "cyberlamp") & (subgroup != "phi")')),
    ])

    for cluster_name, cni in cluster_node_info.items():
        print('>>   Working on {} cluster'.format(cluster_name))
        # TODO: cyberlamp has different CPUs available on the 1-GPU nodes for
        # GPU vs. non-GPU jobs. Knowing which cores are "taken" (or if it's
        # just a count, not particular cores that are dedicatd to job type) is
        # not possible as the pbsnodes parsing script stands (might need data
        # from another source about actual jobs that are running on each node
        # and what they call out).

        sc = slot_cores[slot_cores <= cni.cores.max()]
        sm = slot_mem[slot_mem <= cni.mem_tot.max()]
        if 'gpu' in cluster_name.lower():
            sg = 1
            sc = sc[sc <= 20]
            sm = sm[sm <= 242]
        else:
            sg = 0

        slots = compute_open_slots(
            node_info=cni,
            slot_cores=sc,
            slot_mem=sm,
            slot_gpus=sg
        )

        with np.errstate(divide='ignore'):
            log_slots = np.log10(slots)
        neginf_mask = np.isinf(log_slots)
        neginf_fill_val = -np.max(log_slots)/6
        log_slots[neginf_mask] = neginf_fill_val

        min_slots = slots.min()
        max_slots = slots.max()

        lin_tickvals = np.array(
            [1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6],
            dtype=int
        )
        lin_tickvals = lin_tickvals[(lin_tickvals > min_slots)
                                    & (lin_tickvals < max_slots)]

        if len(lin_tickvals) < 3:
            order = np.floor(np.log10(min_slots))
            lin_tickvals = np.arange(10**np.floor(np.log10(min_slots)),
                                     max_slots + 1,
                                     10**order,
                                     dtype=int)

        lin_tickvals = lin_tickvals.tolist() + [max_slots]
        lin_tickvals = np.array(sorted(set(lin_tickvals)), dtype=int)

        tickvals = np.log10(lin_tickvals)

        lin_tickvals = sorted(set([0] + lin_tickvals.tolist()))
        tickvals = [neginf_fill_val] + tickvals.tolist()

        ticktext = ['{:d}'.format(tv) for tv in lin_tickvals]

        text = np.empty_like(slots, dtype=str).tolist()
        for (core_i, cores), (mem_i, mem) in product(enumerate(sc),
                                                     enumerate(sm)):
            if cores == 1:
                ct = 'core'
            else:
                ct = 'cores'
            val = slots[core_i, mem_i]
            if val == 1:
                slt = 'slot'
            else:
                slt = 'slots'
            text[core_i][mem_i] = (
                '{:d} {:s} & {:d} GiB mem : {:d} {:s}'
                .format(cores, ct, mem, val, slt)
            )

        trace = graph_objs.Heatmap(
            z=log_slots,
            x=[str(m) + ' GiB' for m in sm],
            y=[str(sc[0]) + ' core'] + [str(c) + ' cores' for c in sc[1:]],
            zsmooth=False,
            xgap=1,
            ygap=1,
            colorscale='Viridis',
            colorbar=dict(
                outlinewidth=0,
                tickvals=tickvals,
                ticktext=ticktext
            ),
            hoverinfo='text',
            text=text,
        )
        data = [trace]
        layout = graph_objs.Layout(
            title='{} Job Slots Available'.format(cluster_name),
            annotations=graph_objs.Annotations([
                graph_objs.Annotation(
                    x=0.5,
                    y=1.07,
                    showarrow=False,
                    text=updated_at,
                    xref='paper',
                    yref='paper'
                ),
            ]),
            xaxis=dict(ticks='', nticks=36),
            yaxis=dict(ticks='')
        )
        fig = graph_objs.Figure(data=data, layout=layout)
        plotly.plot(fig, filename=cluster_name, auto_open=False)

    if DEBUG:
        print('>>   plot_slot_avail:', time.time() - t0)


if __name__ == '__main__':
    node_info, gpu_info, sec_since_epoch = get_info()
    plot_slot_avail(node_info, sec_since_epoch)
