import numpy as np
import os

pwd=os.getcwd()

for i in range(1,31):   
    os.chdir(pwd+f"/{int(i)}")
    os.system("rm dados_* visc_time2*")
    os.chdir(pwd)


    
    
    
    
    
    





