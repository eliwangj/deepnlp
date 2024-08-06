import sys
sys.path.append("../..")

from DeepNLP.utils.args_utils import get_args
from DeepNLP.model.DSeg import DSeg


def train():
    parser = get_args()

    args = parser.parse_args()

    DSeg.fit(args)


if __name__ == '__main__':
    train()