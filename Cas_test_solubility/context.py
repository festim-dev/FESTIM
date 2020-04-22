import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

#print(sys.path) # to check if the "fenics" folder is located in one of the sys.path cell

import FESTIM
