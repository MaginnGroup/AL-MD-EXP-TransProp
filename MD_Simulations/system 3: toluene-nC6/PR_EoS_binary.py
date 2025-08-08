import numpy as np
import matplotlib.pyplot as plt

# INPUTS

P = 101325 # [Pa]
T = 318.0 #[K]
x1 = 0.95 
R = 8.314  #  [J/(mol*K)]
kij = 3.22e-4 
component1 = 'toluene'
component2 = 'hexane'

#DATABANK

components = {
    "toluene": {"Tc": 591.8, "Pc": 4100.04e3, "omega": 0.2596},  
    "hexane": {"Tc": 507.9, "Pc": 3031.62e3, "omega": 0.3007}
     }
P_vector = np.arange(1e5, 100e5, 0.1e5)
x1_vector = np.arange(0.01, 0.99, 0.01) 
T_vector = np.arange(280, 400, 1) 

# FUGACITY CALCULATION

def calculate_f(x1, components, T, P):
    
    x2 = 1 - x1

    def calc_params(component):
        Tc, Pc, omega = component["Tc"], component["Pc"], component["omega"]
        a = 0.45724 * R**2 * Tc**2 / Pc
        b = 0.07780 * R * Tc / Pc
        alpha = (1 + (0.37464 + 1.54226 * omega - 0.26992 * omega**2) * (1 - np.sqrt(T / Tc)))**2
        return a * alpha, b
    
    # parameters
    params = {name: calc_params(components[name]) for name in components}
    a1, b1 = params[component1]
    a2, b2 = params[component2]

    # mixing rules
    amix = x1**2 * a1 + x2**2 * a2 + 2 * x1 * x2 * np.sqrt(a1 * a2) * (1 - kij)
    bmix = x1 * b1 + x2 * b2

    # cubic coefficients for Z
    A = amix * P / (R * T)**2
    B = bmix * P / (R * T)
    coeffs = [1.0, -(1.0 - B), A - 2.0 * B - 3.0 * B**2, -(A * B - B**2 - B**3)]
    
    # solve for Z
    roots = np.roots(coeffs)
    Z_liquid = min(roots[np.isreal(roots)])  # lower real result for liquid phase 

    def fugacity_coeff_liquid(a_i, b_i, x_i, i):
        bi_bmix = b_i / bmix
        term1 = bi_bmix * (Z_liquid - 1)
        term2 = -np.log(Z_liquid - B)
        
        if i == 1:
            term3 = -A / (2 * np.sqrt(2) * B) * (2 * (x1 * np.sqrt(a1 * a_i) + x2 * np.sqrt(a2 * a_i) * (1-kij)) / amix - bi_bmix)
        if i == 2:
            term3 = -A / (2 * np.sqrt(2) * B) * (2 * (x1 * np.sqrt(a1 * a_i) * (1-kij) + x2 * np.sqrt(a2 * a_i)) / amix - bi_bmix)
                
        term3 *= np.log((Z_liquid + (1 + np.sqrt(2)) * B) / (Z_liquid + (1 - np.sqrt(2)) * B))
        ln_phi = term1 + term2 + term3
        return np.exp(ln_phi)
    
    # fugacity coefficients
    phi1_liquid = fugacity_coeff_liquid(a1, b1, x1, i=1)
    phi2_liquid = fugacity_coeff_liquid(a2, b2, x2, i=2)

    # fugacity
    f1 = phi1_liquid * P * x1
    f2 = phi2_liquid * P * x2

    return f1, f2

# THERMODYNAMIC FACTOR CALCULATION  

def d_ln_fi_d_ln_xi(x1, components, T, P, delta=1e-5):
    
    f1, f2 = calculate_f(x1, components, T, P)
    f1_plus, f2_plus = calculate_f(x1 + delta, components, T, P)
    
    ln_f1 = np.log(f1)
    ln_f1_plus = np.log(f1_plus)
    ln_f2 = np.log(f2)
    ln_f2_plus = np.log(f2_plus)
    
    d_ln_f1_d_ln_x1 = (ln_f1_plus - ln_f1) / np.log((x1 + delta) / x1)
    d_ln_f2_d_ln_x2 = (ln_f2_plus - ln_f2) / np.log((1 - x1 - delta) / (1 - x1))
    
    return d_ln_f1_d_ln_x1,d_ln_f2_d_ln_x2


# FINAL RESULTS

derivatives = d_ln_fi_d_ln_xi(x1, components, T, P)
print(f"Gamma_11 = {np.mean(derivatives)}")




