import sys
from agents import utils
from r_cifar_utils import get_cifar_experience

if __name__ == '__main__':
    utils.fix_seeds()
    # Load cmd arguments
    args = utils.get_args(sys.argv[1:])
    print(args)
    generic_scenario = get_cifar_experience()

    print(generic_scenario.n_experiences)

