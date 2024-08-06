import sys
sys.path.append("../..")

from DeepNLP.utils.args_utils import get_args
from DeepNLP.model.DSRL_dep_v4 import DSRL


def train():
    parser = get_args()

    args = parser.parse_args()

    DSRL.fit(args)


if __name__ == '__main__':
    train()