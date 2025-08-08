import numpy as np
from random import *
import os

T = 313.16
a = 1
b = 150

pwd=os.getcwd()

for i in range(1,31):
   
    os.system(f"mkdir {int(i)}")
    os.system("cp in.run "+pwd+f"/{int(i)}/in.run")
    os.system("cp viscosity2.py "+pwd+f"/{int(i)}/viscosity2.py")
    os.system("cp diffusivity.py "+pwd+f"/{int(i)}/diffusivity.py")    
     
    os.chdir(pwd+f"/{int(i)}")
    with open('in.run', 'r') as file :
        filedata = file.read()
    file_fixed1 = filedata.replace('xst1', f'{T}')
    file_fixed2 = file_fixed1.replace('xst2', f'{randint(0,100000)}')
    file_fixed3 = file_fixed2.replace('xsta', f'{int(a)}')
    file_fixed4 = file_fixed3.replace('xstb', f'{int(b)}')
    with open('in.run', 'w') as file:
        file.write(file_fixed4)  
    os.chdir(pwd)


    
    
    
    
    
    





