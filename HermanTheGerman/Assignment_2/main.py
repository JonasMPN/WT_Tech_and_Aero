import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton

do = {
    "1 and 2": False,
    "3": False,
    "4": True
}


if do["1 and 2"]:
    def calc(P_mech: float, generator_frequency: float, magnetic_flux: float, resistance: float, reactance: float):
        I_a = P_mech/(3*generator_frequency*magnetic_flux*np.cos(np.arctan(generator_frequency*reactance/resistance)))
        P_loss = I_a**2*(resistance**2+(frequency_grid*reactance)**2)
        V_g = (P_mech-P_loss)/I_a
        return I_a, V_g, P_loss

if do["3"] or do["4"]:
    def refer_impedances_to_primary(transformer_ratio: float, *args):
        return [arg*transformer_ratio**2 for arg in args]

    def refer_voltages_to_primary(transformer_ratio: float, *args):
        return [arg*transformer_ratio for arg in args]

    def refer_currents_to_primary(transformer_ratio: float, *args):
        return [arg/transformer_ratio for arg in args]


    def calculations(generator_power: float,
                     reactive_power_fac: float,
                     V_grid_ref_prim: float,
                     grid_frequency: float,
                     core_inductance: complex,
                     cable_capacity_ref_prim: complex,
                     impedance_primary: complex,
                     impedance_sec_ref_prim: complex):
        C_ref_prim = cable_capacity_ref_prim
        L_t = core_inductance
        omega = grid_frequency
        Z_1 = impedance_primary
        Z_2_ref_prim = impedance_sec_ref_prim
        power = generator_power*(1+reactive_power_fac*1j)

        imp_cap = 1j*omega*C_ref_prim
        imp_core = 1j*omega*L_t
        I_sqr_fac = (Z_1+Z_2_ref_prim/(1+Z_2_ref_prim/imp_core))
        p = V_grid_ref_prim*(imp_cap*imp_sec_ref_prim+1-(imp_cap*(imp_core+imp_sec_ref_prim)+1)/(imp_core/Z_2_ref_prim+1))/I_sqr_fac
        q = -power/I_sqr_fac
        sqrt = np.sqrt(p**2/4-q)
        I_1 = [-p/2+sqrt, -p/2-sqrt]
        power_test = False
        I_poc_ref_prim, V_B, I_t, I_2 = list(), list(), list(), list()
        for I in I_1:
            I_poc_rp = (I-V_grid_ref_prim*(imp_cap*(1+imp_sec_ref_prim/imp_core)+1/imp_core))/(1+Z_2_ref_prim/imp_core)
            I_poc_ref_prim.append(I_poc_rp)
            I_T =((imp_cap*V_grid_ref_prim+I_poc_rp)*Z_2_ref_prim+V_grid_ref_prim)/imp_core
            i_2 = I-I_T
            V_B.append(power/I)
            I_t.append(I_T)
            I_2.append(i_2)
            if power_test:
                print("Supplied power by VSC-B minus power drawn from all components: ",
                      power-I**2*impedance_primary-I_T**2*imp_core-i_2**2*impedance_sec_ref_prim-(i_2-I_poc_rp)**2*imp_cap-I_poc_rp*V_grid_ref_prim)
        return I_poc_ref_prim, I_1, V_B, I_t, I_2


    generator_power = 10e6
    V_grid = 33e3
    frequency_grid = 50
    transformer_ratio = 4/33
    cable_length = 65
    R_cable = 0.1*cable_length
    inductance_cable = 0.5e-3*cable_length
    capacitance_cable = 0.1e-6*cable_length

    R_cable_ref_prim, ind_cable_ref_prim, cap_cable_ref_prim = refer_impedances_to_primary(transformer_ratio,
                                                                                           R_cable,
                                                                                           inductance_cable,
                                                                                           capacitance_cable)
    V_grid_ref_prim, = refer_voltages_to_primary(transformer_ratio, V_grid)
    imp_sec_ref_prim = 25e-3+R_cable_ref_prim+1j*(0.6e-3+ind_cable_ref_prim)
    if do["3"]:
        I_poc_ref_prim, I_1, V_B, I_t, I_2_ref_prim = calculations(generator_power= generator_power,
                                                                   reactive_power_fac=0.2,
                                                                   V_grid_ref_prim= V_grid_ref_prim,
                                                                   grid_frequency= frequency_grid,
                                                                   core_inductance= 0.046,
                                                                   cable_capacity_ref_prim= cap_cable_ref_prim,
                                                                   impedance_primary= 25e-3+1j*0.6e-3,
                                                                   impedance_sec_ref_prim= imp_sec_ref_prim)
        I_poc, I_2 = list(), list()
        for i_poc_ref_prim, i_2_ref_prim in zip(I_poc_ref_prim, I_2_ref_prim):
            i_poc, i_2 = refer_currents_to_primary(1/transformer_ratio, i_poc_ref_prim, i_2_ref_prim)
            I_poc.append(i_poc)
            I_2.append(i_2)
        print(-np.asarray(I_poc)*V_grid)
    if do["4"]:
        def residue(reactive_power_fac: float) -> float:
            I_poc_ref_prim, _ = calculations(generator_power=generator_power, reactive_power_fac=reactive_power_fac,
                                        V_grid_ref_prim=V_grid_ref_prim, grid_frequency=frequency_grid, core_inductance=0.046,
                                        cable_capacity_ref_prim=cap_cable_ref_prim, impedance_primary=25e-3 + 1j * 0.6e-3,
                                        impedance_sec_ref_prim=imp_sec_ref_prim)[0]
            return I_poc_ref_prim.imag
        RPF = newton(residue, x0=-0.8175)
        I_poc_ref_prim, = calculations(generator_power=generator_power, reactive_power_fac=RPF,
                                        V_grid_ref_prim=V_grid_ref_prim, grid_frequency=frequency_grid, core_inductance=0.046,
                                        cable_capacity_ref_prim=cap_cable_ref_prim, impedance_primary=25e-3 + 1j * 0.6e-3,
                                        impedance_sec_ref_prim=imp_sec_ref_prim)[0]
        i_poc = refer_currents_to_primary(1/transformer_ratio, I_poc_ref_prim)
        print("Reactive power coefficient: ", RPF, "yields I_poc ", i_poc)
