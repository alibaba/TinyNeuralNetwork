#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ "$OSTYPE" == "msys" ]]; then
    DIR="$(cygpath -d $DIR)"
    SEP=";"
    DELIM="\\"
else
    SEP=":"
    DELIM="/"
fi

export PYTHONPATH=${PYTHONPATH}${SEP}${DIR}${SEP}${DIR}${DELIM}examples
