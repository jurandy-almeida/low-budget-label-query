#!/bin/bash

LOGS="$@"

printf "NAME\tBEST\tLAST\n"
for LOG in $LOGS
do
	NAME=`basename "$LOG" .log`
	BEST=`grep ' * Prec@1' "$LOG"  | sed -e 's%^.* %%g'  | sort -n | tail -n 1 | tr '.' ','`
	LAST=`tail -n 1 "$LOG" | sed -e 's%^.* %%g' | tr '.' ','`
	printf "%s\t%s\t%s\n" "$NAME" "$BEST" "$LAST"
done
