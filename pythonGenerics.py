import re, os, sys, time


def timestamp(d=True, t=True, tz=True, utc=False, winsafe=False):
    '''Simple utility to print out a time, date, or time&date stamp,
    with some reconfigurability for commonly-used options. Default is in
    ISO8601 format without colons separating hours, min, and sec to avoid
    file naming issues.

    Options:
        d          print date (default: True)
        t          print time (default: True)
        tz         print timezone offset from UTC (default: True)
        utc        print time/date in UTC (default: False)
        winsafe    omit colons between hours/minutes (default: True)

    '''
    if utc:
        timeTuple = time.gmtime()
    else:
        timeTuple = time.localtime()

    dts = ""
    if d:
        dts += time.strftime("%Y-%m-%d", timeTuple)
        if t:
            dts += "T"
    if t:
        if winsafe:
            dts += time.strftime("%H%M%S", timeTuple)
        else:
            dts += time.strftime("%H:%M:%S", timeTuple)

        if tz:
            if utc:
                if winsafe:
                    dts += time.strftime("+0000")
                else:
                    dts += time.strftime("+00:00")
            else:
                offset = time.strftime("%z")
                if not winsafe:
                    offset = offset[:-2:] + ":" + offset[-2::]
                dts += offset
    return dts


def findFiles(directory, regex):
    if isinstance(regex, (str, unicode)):
        regex = re.compile(regex)
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if len(regex.findall(basename)) > 0:
                filename = os.path.join(root, basename)
                yield (root, basename) #filename


def wstdout(x):
    sys.stdout.write(x)
    sys.stdout.flush()


def wstderr(x):
    sys.stderr.write(x)
    sys.stderr.flush()
