from BEM import BEM
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from helper_functions import Helper
helper = Helper()
res = 50
v_min, v_max, rpm_min, rpm_max = 10, 25, 6, 9.6
omega_min, omega_max = rpm_min*np.pi/30, rpm_max*np.pi/30
rotor_radius = 89.17
bem = BEM("airfoil_data",
          "blade_data_new.txt",
          "combined_data_new.txt",
          {"c_l": ["rel_thickness", "alpha"], "c_d": ["rel_thickness", "alpha"]})
bem.set_constants(rotor_radius=rotor_radius,
                  n_blades=3,
                  v0=10,
                  air_density=1.225)
tip_speed_ratios, pitch_angles, c_Ps = bem.optimise(tpr_interval=(omega_min*rotor_radius/v_max, omega_max*rotor_radius/v_min),
                                                    pitch_interval=(-2,5), resolution=res)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(tip_speed_ratios,pitch_angles, c_Ps, cmap=cm.coolwarm, linewidth=0, antialiased=False)
helper.handle_axis(ax, r"$c_p$", x_label="TSR", y_label="pitch", z_label="c_p")
plt.close(helper.handle_figure(fig, f"data/surface_{res}.png"))

fig, ax = plt.subplots()
ax.contourf(tip_speed_ratios, pitch_angles, c_Ps)
helper.handle_axis(ax, r"$c_p$", x_label="TSR", y_label="pitch")
plt.close(helper.handle_figure(fig, f"data/contour_{res}.png"))