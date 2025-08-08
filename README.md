## Electronic Supporting File Repository

**Authors:** Gabriela B. Correa, Eliseo Marin-Rimoldi, Frederico W. Tavares, and Edward J. Maginn

This repository contains the supporting files for the article:  
**_Active Learning for Transport Property Prediction in CO₂–Hydrocarbon Systems: A Multi-Fidelity Approach Integrating Molecular Dynamics and Experiments_**

The repository is divided into two main folders. Each main folder is organized into subdirectories corresponding to the binary and ternary systems studied: **CO₂/n-heptane**, **CO₂/benzene**, **toluene/n-hexane**, and **CO₂/ethanol/dibenzyl ether**.

1. **`Active_Learning/`**
   Contains scripts and datasets used in the active learning workflows:
   - Experimental and MD datasets  
   - Python scripts for Multi-fidelity Gaussian Process (integrating MD and experimental data) 
   - Python scripts for Single-fidelity Gaussian Process (MD-only or experimental-only)

2. **`MD_Simulations/`**
   Contains input and analysis files for molecular dynamics simulations:
   - Force field topology and parameter files  
   - LAMMPS simulation scripts  
   - Python scripts for calculating mutual diffusivity and viscosity
