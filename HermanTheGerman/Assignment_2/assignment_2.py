import numpy as np


class Assignment2():
    @staticmethod
    def gen_output(P_mech: float,
                   generator_angular_frequency: float,
                   magnetic_flux: float,
                   resistance: float,
                   reactance: float):
        delta = np.arcsin(2/3*P_mech*reactance/(generator_angular_frequency*magnetic_flux**2))/2
        I_g = P_mech/(3*generator_angular_frequency*magnetic_flux*np.cos(delta))
        V_g = generator_angular_frequency*magnetic_flux*np.cos(delta)-I_g*resistance
        P_loss = I_g**2*resistance
        E_a = V_g+I_g*(resistance+1j*generator_angular_frequency*reactance)
        return I_g, V_g, P_loss, E_a

    @staticmethod
    def refer_impedances_to_primary(transformer_ratio: float, *args):
        return [arg*transformer_ratio**2 for arg in args]

    @staticmethod
    def refer_voltages_to_primary(transformer_ratio: float, *args):
        return [arg*transformer_ratio for arg in args]

    @staticmethod
    def refer_currents_to_primary(transformer_ratio: float, *args):
        return [arg/transformer_ratio for arg in args]

    @staticmethod
    def calculations(input_power: float,
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
        power = input_power

        imp_cap = 1j*omega*C_ref_prim
        imp_core = 1j*omega*L_t
        I_sqr_fac = (Z_1+Z_2_ref_prim/(1+Z_2_ref_prim/imp_core))
        p = V_grid_ref_prim*(imp_cap*Z_2_ref_prim+1-(imp_cap*(imp_core+Z_2_ref_prim)+1)/(imp_core/Z_2_ref_prim+1))/I_sqr_fac
        q = -power/I_sqr_fac
        sqrt = np.sqrt(p**2/4-q)
        I_1 = [-p/2+sqrt, -p/2-sqrt]
        power_test = False
        I_poc_ref_prim, V_B, I_t, I_2 = list(), list(), list(), list()
        for I in I_1:
            I_poc_rp = (I-V_grid_ref_prim*(imp_cap*(1+Z_2_ref_prim/imp_core)+1/imp_core))/(1+Z_2_ref_prim/imp_core)
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