#!/bin/sh

if [[ "$1" == "-q" ]]; then
	QUIET="y";
else
	QUIET="n";
fi

# Information on total cores from http://rcc.its.psu.edu/resources/hpc/ (and private comm. for xv)
# updating with info from /usr/moab/bin/showq <- ?
case ${HOSTNAME/\.*/} in
	lionxg)
		TOTAL_CORES=3072;
		;;
	lionxf)
		TOTAL_CORES=2148;
		#TOTAL_CORES=2208;
		;;
	lionxi)
		TOTAL_CORES=232;
		#TOTAL_CORES=512;
		;;
	lionxj)
		TOTAL_CORES=904;
		#TOTAL_CORES=1152;
		;;
	lionxh)
		TOTAL_CORES=272;
		#TOTAL_CORES=336;
		;;
	lionxv)
		TOTAL_CORES=1920;
		;;
esac

showq | awk -v user=${USER} '
		BEGIN{
			reach_blocked=0;
		}
		{
			if( $1=="blocked" ){
				reach_blocked=1;
			}
			if( $2=="active" && $3=="jobs" ){
				total_cores=$6;
				next;
			}
			if( !reach_blocked && ($4*2)/2 == $4)
				print $3" "$2" "$4
			if( reach_blocked && (($4*2)/2 == $4) && user==$2)
				print "Hold "$2" "$4
		}
		END{
			printf "TOTAL_CORES %d\n",total_cores
		}
	' | sort | \
	awk -v total_cores=${TOTAL_CORES} -v quiet=${QUIET} -v user=${USER} '
		BEGIN{
			prev="";
			cores_running=0;
			cores_queued=0;
			user_running=0;
			user_queued=0;
			user_hold=0;
		}
		{
			if($1=="TOTAL_CORES"){
				total_cores=$2;
				next;
			}
			if(user==$2){
				cur="\033[1;31m"$1" "$2"\033[0m";
				if($1=="Idle"){
					user_queued+=1;
				}
				else if($1=="Running"){
					user_running+=1;
				}
				else if($1=="Hold"){
					user_hold+=1;
				}
			}
			else{
				cur=$1" "$2;
			}
			if(prev!=cur){
				if(prev!="" && quiet!="y") printf "%6d %s\n", counter, prev;
				prev=cur;
				counter=$3;
			}
			else{
				counter+=$3;
			}
			if($1=="Idle"){
				cores_queued+=$3;
			}
			else if($1=="Running"){
				cores_running+=$3;
			}
		}
		END{
			if(quiet!="y") printf "%6d %s\n", counter, cur;
			else{
				printf "%8s user has \033[1;31m%6d\033[0m jobs running; %6d jobs queued; %6d jobs on hold\n",
					user, user_running, user_queued, user_hold;
			}
			printf "++++++ Summary => %6d (of %6d) cores running; %6d req. core queued\n",
				cores_running,total_cores,cores_queued;
		}
	'
