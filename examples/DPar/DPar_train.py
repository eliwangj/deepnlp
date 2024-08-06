import sys
sys.path.append("../..")

from DeepNLP.utils.args_utils import get_args
from DeepNLP.model.DPar import DPar


def train():
    parser = get_args()
    args = parser.parse_args()
    DPar.fit(args)

if __name__ == '__main__':
    train()