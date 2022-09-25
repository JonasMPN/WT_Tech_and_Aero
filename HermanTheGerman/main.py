from BEM import BEM
import numpy as np
import matplotlib.pyplot as plt

rpm = 9.6
bem = BEM("airfoil_data",
          "blade_data.txt",
          "combined_data.txt",
          {"c_l": ["rel_thickness", "alpha"], "c_d": ["rel_thickness", "alpha"]})
bem.set_constants(v0=11.4,
                  omega=rpm/30*np.pi,
                  rotor_radius=89.17,
                  n_blades=3,
                  air_density=1.225)
radius, N, T = bem.solve(pitch=0)
plt.plot(radius, N)
plt.plot(radius, T)
plt.grid()
plt.show()