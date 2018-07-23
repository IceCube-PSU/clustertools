#!/usr/bin/env python
import multiprocessing
import time
import commands

jobs = commands.getstatusoutput('qselect -u $USER -s R')
jobs = jobs[1].split()

def mp_worker(job):
    print " Deleting %s"%job
    out = commands.getstatusoutput('qdel %s'%job)

def mp_handler():
    p = multiprocessing.Pool(100)
    p.map(mp_worker, jobs)

if __name__ == '__main__':
    mp_handler()
