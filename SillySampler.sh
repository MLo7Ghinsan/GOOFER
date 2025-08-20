#!/bin/sh
RELPATH="SillySampler.py"

ABSPATH=$(cd "$(dirname "$0")"; pwd -P)
ABSPATH="$ABSPATH/$RELPATH"
if [[ ! -x "$ABSPATH" ]]
then
    chmod +x "$ABSPATH"
fi
exec /usr/local/bin/python3 $ABSPATH "$@"