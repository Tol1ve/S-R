#to train a total MLP with latent codes
#to train a set of MLPs for each subject
#import sys
#from os.path import abspath, join, dirname
#sys.path.insert(0, join(abspath(dirname(__file__)), 'src'))
#it can add the filepath to pythonenvpath

import numpy as np
import nesvor_mod.cli.commands

from nesvor_mod.cli.main import main
#import err: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS' from 'charset_normalizer.constant' 
#solved:conda install -c conda-forge charset-normalizer

#args build

#entry point
if __name__ == "__main__":
    main()