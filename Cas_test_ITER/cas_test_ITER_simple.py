# Bonjour, ce script est à executer en se trouvant dans
# FESTIM_4_JONATHAN/Cas_test_ITER
# Pour executer run : python3 cas_test_ITER_simple.py
# Ce script produira dans
# FESTIM_4_JONATHAN/Cas_test_ITER/results/[nombre_de_pieges]
# des fichiers XDMF et un fichier .csv


from context import FESTIM
from parameters import parameters

if __name__ == "__main__":
    # Run
    FESTIM.generic_simulation.run(parameters, log_level=40)
