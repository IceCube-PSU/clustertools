# Script to run cluster jobs inside CVMFS evironement (e.g. parrot on ACI)
# originally from J. Lanfranchi
#
# Usage:
#
# in job script add on top `source THISFILE.sh`
# and then execute your command for example as:
# 
# i3_run py2-v3.1.1 "$ENV_SHELL" 'python' 'my_python_script.py' --arg1 blah
#
# (change accordingly)
#


#==============================================================================
# Setup environment similar to what's done in .bashrc
#
# (Also clears out PATH and PYTHONPATH so that user's settings don't interfere
# with this script)
#==============================================================================

export PARROT_RUN="/gpfs/group/dfc13/default/cctools/bin/parrot_run"
export PARROT_CVMFS_ALIEN_CACHE="/gpfs/group/dfc13/default/cache/"
export CVMFS="/cvmfs/icecube.opensciencegrid.org"
ENV_SHELL="/gpfs/group/dfc13/default/build/i3/oscNext/env-shell.sh"

#==============================================================================
# Define functions whereby CVMFS is accessible
# via the Parrot user-space tool (and is intended--but not tested--to also work
# with a proper native installation of CVMFS)
#==============================================================================

function init_i3_env () {
        echo $*
        py2_vx=$1
        shift
        env_shell=$1
        shift
        if [ -x "$CVMFS/${py2_vx}/setup.sh" ]
        then
                eval $( $CVMFS/${py2_vx}/setup.sh )
        else
                echo "ERROR: $CVMFS/${py2_vx}/setup.sh not found or not executable!"
        fi

        if [ -x "${env_shell}" ]
        then
                ${env_shell} "$*"
        else
                echo "ERROR: ${env_shell} not found or not executable!"
        fi
}
export -f init_i3_env

function i3_run () {
        if [ ! -d "$CVMFS" ]
        then
                if [ -x "`which cvmfs_config 2>/dev/null`" ]
                then
                        cvmfs_config probe
                        /bin/sh --norc -c "'init_i3_env' $*"
                elif [ ! -d "$CVMFS" -a -x "$PARROT_RUN" ]
                then
                        HTTP_PROXY="cache01.hep.wisc.edu:3128" "$PARROT_RUN" /bin/sh --norc -c "'init_i3_env' $*"
                fi
        fi
}
export -f i3_run
