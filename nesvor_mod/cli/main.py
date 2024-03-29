"""nesvor entrypoints"""


import sys
import string
import logging
from .parsers import main_parser
import json


def main() -> None:
    
    parser, subparsers = main_parser()
    if len(sys.argv) == 1:
        # read the settings from config file if no args provided and contain args in config file
        #  todo:what if the root dir change?
        with open('./S-R/config/config.json', 'r') as file:
            commmands = json.load(file)['configurations']
        for commmand in commmands:
            if len(sys.argv) > 1:
                break
            if isinstance(commmand, dict) and "commmand" in commmand:
                # if there are various arg config, the default is the first dict.
                sys.argv.append(commmand["commmand"])
                for key, value in commmand.items():
                    if key != "commmand":
                        sys.argv.append(key)
                        if isinstance(value, list):
                            for v in value:
                                sys.argv.append(v)
                        else:
                            sys.argv.append(value)
            else:
                logging.warning(
                    "find a non-dict or no-commmand arg setting in config file, will ignore it\
                    ,please make sure your config with the correct format")
        # print help if no args are provided and no args in config file
        
        if len(sys.argv) == 1:
            parser.print_help(sys.stdout)
            return
    if len(sys.argv) == 2:
        if sys.argv[-1] in subparsers.choices:
            subparsers.choices[sys.argv[-1]].print_help(sys.stdout)
            return
    
    # parse args
    args = parser.parse_args(args=sys.argv[1:])
    
    run(args)


def run(args) -> None:
    import torch
    from . import commands
    from .. import utils

    # setup logger
    if args.debug:
        args.verbose = 2
    utils.setup_logger(args.output_log, args.verbose)
    # setup device
    if args.device >= 0:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")
        logging.warning(
            "NeSVoR is running in CPU mode. The performance will be suboptimal. Try to use a GPU instead."
        )
    # setup seed
    utils.set_seed(args.seed)

    # execute command
    command_class = "".join(string.capwords(w) for w in args.command.split("-"))
    getattr(commands, command_class)(args).main()


if __name__ == "__main__":
    main()
