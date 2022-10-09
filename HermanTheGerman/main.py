from BEM import BEM
from matplotlib import cm
import numpy as np
from helper_functions import Helper
from data_handling import BemData
helper = Helper()
bem_data = BemData("data")

do = {
    "calculate_cp_max": False,
    "calculate_pitch_curve": True,
    "contourf": False,
    "surface": False,
    "plot_pitch_curve": True
}
res = 50
pitch_step_size = 0.01

bem = BEM("airfoil_data",
          "data",
          "blade_data_new.txt",
          "combined_data_new.txt",
          {"c_l": ["rel_thickness", "alpha"], "c_d": ["rel_thickness", "alpha"]})
v_min, v_max, rpm_min, rpm_max = 4, 25, 6, 9.6
omega_min, omega_max = rpm_min*np.pi/30, rpm_max*np.pi/30
rotor_radius = 89.17
bem.set_constants(rotor_radius=rotor_radius,
                  n_blades=3,
                  v0=10,
                  air_density=1.225)
if do["calculate_cp_max"]:
    bem.optimise(tsr_interval=(omega_min*rotor_radius/v_max, omega_max*rotor_radius/v_min), pitch_interval=(-2,5),
                 resolution=res)

if do["calculate_pitch_curve"]:
    bem.pitch_curve(rated_power=10.64e6, wind_speeds=(v_min,v_max,res),
                    pitch_step_size=pitch_step_size, tsr_optimum=8, pitch_optimum=0)

if do["contourf"]:
    contourf_kwargs = {
        "levels": [0,0.1,0.2,0.3,0.4,0.44,0.45,0.46, 0.465, 0.5],
        "cmap": cm.RdYlGn}
    axes_kwargs = {
        "title": r"$c_P$",
        "x_label": "TSR",
        "y_label": "pitch",
        "font_size": 20}
    figure_kwargs = {
        "size": (18.5, 10)}
    bem_data.contourf(resolution=res,
                      contourf_kwargs=contourf_kwargs,
                      axes_kwargs=axes_kwargs,
                      figure_kwargs=figure_kwargs)

if do["surface"]:
    surface_kwargs = {
        "linewidth": 0,
        "cmap": cm.RdYlGn,
        "antialiased": False}
    axes_kwargs = {
        "title": r"$c_P$",
        "x_label": "TSR",
        "y_label": "pitch",
        "z_label": r"$c_P$"}
    figure_kwargs = {
        "size": (18.5, 10)}
    bem_data.surface(resolution=res,
                     surface_kwargs=surface_kwargs,
                     axes_kwargs=axes_kwargs,
                     figure_kwargs=figure_kwargs)

if do["plot_pitch_curve"]:
    bem_data.pitch_curve(pitch_step_size=pitch_step_size)

