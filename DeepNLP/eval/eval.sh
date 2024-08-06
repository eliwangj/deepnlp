#!/bin/bash
#SBATCH -J YY_1
CONLL_PATH=$1"/eval/srlconll-1.1"
# $1 eval_path
# $2 gold file
# $3 predict file

export PERL5LIB="${CONLL_PATH}/lib:$PERL5LIB"
export PATH="${CONLL_PATH}/bin:$PATH"

perl "${CONLL_PATH}/bin/srl-eval.pl" $2 $3
# perl "${CONLL_PATH}/bin/eval_09.pl" $2 $3

