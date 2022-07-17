#!/bin/bash

LOGS="$@"

function compute_average()
{
	LOGS="$@"
	for LOG in $LOGS
	do
        	NAME=`basename "$LOG" .log`
	        BEST=`grep ' * Prec@1' "$LOG" | sed -e 's%^.* %%g'  | sort -n | tail -n 1`
        	LAST=`grep ' * Prec@1' "$LOG" | tail -n 1 | sed -e 's%^.* %%g'`
	        printf "%s\t%s\t%s\n" "$NAME" "$BEST" "$LAST"
	done | awk -F '\t' '{ best += $2; last += $3 }END{ printf "%0.3f\t%0.3f\n", best / NR, last / NR }' | tr '.' ','
}

function print_result()
{
	LOG="$1"
	BEST=`grep ' * Prec@1' "$LOG" | sed -e 's%^.* %%g'  | sort -n | tail -n 1 | tr '.' ','`
	LAST=`grep ' * Prec@1' "$LOG" | tail -n 1 | sed -e 's%^.* %%g' | tr '.' ','`
	printf "%s\t%s\n" "$BEST" "$LAST"
}

printf "NAME\tBEST\tLAST\n"
for LOG in $LOGS
do
	NAME=`basename "$LOG" .log`
	if [[ "$NAME" == *_? ]]
	then
		if [[ "$NAME" == *_0 ]]
		then
			printf "%s\t" "${NAME%_?}"
			compute_average "${LOG/%_0.log/_*.log}"
		fi
	else
		printf "%s\t" "$NAME"
		print_result "$LOG"
	fi
done
