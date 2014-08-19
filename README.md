clustertools
============

Generic tools for working with clusters

* Server access
  - hammer
  - lion

* Server Status (Note that all of the below require pdsh be installed on your local PC)
  - hammerload: Report the load from all hammer nodes. Requires "loadusersuptime" script be in your $PATH on hammer. Arguments are your ssh username and (optionally) the timeout in sec.
    + loadusersuptime: place this somewhere in your $PATH on hammer for hammerload to call
  - hammertop: top 5 processes sorted by %CPU and by %memory on all hammer nodes. Requires "mytop" script be in the $PATH on hammer. Arguments are your ssh username and (optionally) the timeout in sec.
  - hammertopuser: top 5 processes by %CPU and %memory used by a given user. Requires "mytopuser" script be in the $PATH on hammer. Arguments are your ssh username and (optionally) the username whose processes you wish to examine. If this last argument is omitted, it defaults to your own username
    + mytop
    + mytopuser
  - hammeroffenders: Reports the user using most CPU resources on each hammer node, and that user using most memory resources on each hammer node. Requires the "hammertop" script on your local machine (and its dependencies on the server). Arguments are your ssh username and (optionally) the timeout in sec.

* Cluster jobs
  - qsub_wrapper.sh
  - qstat_wrapper.sh
  - check_queue.sh

* File utilities
  - diff_summary
  - disable_svn: rename all .svn directories beneath the current directory to trick subversion into thinking this is not a repository

* Python utilities
  - pythonGenerics.py: a few functions that are generically handy:
    + timestamp: function to produce a rational (read: ISO 8601-like) date and/or timestamp easily
    + findFiles: finds all files in a directory that match a regex
    + wstdout: writes a message (without an end-line character) to stdout and flushes the message for immediate display to the user
    + wstderr: writes a message (without an end-line character) to stderr and flushes the message for immediate display to the user
