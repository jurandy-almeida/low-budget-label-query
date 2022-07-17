#!/bin/bash

LOGS="$@"

for LOG in $LOGS
do
	NAME=`basename "$LOG" .log`
	BEST=`grep ' * Prec@1' "$LOG" | sed -e 's%^.* %%g'  | sort -n | tail -n 1`
	LAST=`grep ' * Prec@1' "$LOG" | tail -n 1 | sed -e 's%^.* %%g'`
	printf "%s\t%s\t%s\n" "$NAME" "$BEST" "$LAST"
done | awk -F '\t' '{ best += $2; last += $3 }END{ printf "%0.3f\t%0.3f\n", best / NR, last / NR }' | tr '.' ','
