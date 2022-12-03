import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import newton
from helper_functions import Helper
from assignment_2 import Assignment2
calculate = Assignment2()
helper = Helper()

do = {
    "1 and 2": False,
    "3": True,
    "4": False,
    "5": False,
}


if do["1 and 2"]:
    rpms = np.arange(2,10)
    power_plot = list()
    power_vals = [90748.5288663250,306276.284923847,725988.230930600,1417945.76353633,
                  2450210.27939077,3890843.17514368,5807905.84744480,8269459.69294386]
    I_gs, V_gs, gen_freqs, E_as, losses = list(), list(), list(), list(), list()
    for rpm, mech_power in zip(rpms, power_vals):
        gen_freq = rpm*160/60
        angular_freq = gen_freq*2*np.pi
        I_g, V_g, P_loss, E_a = calculate.gen_output(P_mech=mech_power, generator_angular_frequency=angular_freq,
                                                magnetic_flux=19.49, resistance=64e-3, reactance=1.8e-3)
        gen_freqs.append(gen_freq)
        I_gs.append(I_g)
        V_gs.append(V_g)
        E_as.append(E_a)
        # E_a.append((I_g*V_g+P_loss)/I_g)
        losses.append(P_loss)
    print(E_as)
    fig, ax = plt.subplots(3,1)
    ax[0].plot(rpms, gen_freqs, lw=3)
    ax[1].plot(rpms, np.asarray(V_gs)/1e3, label=r"V_g", lw=3)
    helper.handle_axis(ax[0], "Angular frequency of the generator", grid=True, legend=False, x_label="n in rpm",
                       y_label=r"$f_e$ in rad/s", font_size=15)
    ax[1].plot(rpms, np.asarray(E_as).real/1e3, label=r"real $E_a$", lw=3)
    ax[1].plot(rpms, np.asarray(E_as).imag/1e3, label=r"imag $E_a$", lw=3)
    helper.handle_axis(ax[1], "Generator voltage and back EMF", grid=True, legend=True, x_label="n in rpm",
                       y_label="Voltage in kV", font_size=15)
    ax[2].plot(rpms, np.asarray(I_gs)/1e3, lw=3)
    helper.handle_axis(ax[2], "Current", grid=True, legend=False, x_label="n in rpm", y_label="Current in kA",
                       font_size=15)
    plt.close(helper.handle_figure(fig, "q1.png", size=(15,10)))

    pd.DataFrame({"n in rpm": rpms,
                  "input power in MW": np.asarray(power_vals)/1e6,
                  "output power in MW": np.asarray(V_gs)*np.asarray(I_gs)*3/1e6,
                  "power loss in kW": 3*np.asarray(losses)/1e3}).to_latex("tab_2", index=False)
    fig, ax = plt.subplots(2,1)
    ax[0].plot(rpms, np.asarray(power_vals)/1e6, lw=3, label="input")
    ax[0].plot(rpms, np.asarray(V_gs)*np.asarray(I_gs)*3/1e6, lw=3, label="output")
    helper.handle_axis(ax[0], title="Powers in the generator circuit", grid=True, x_label="n in rpm",
                       y_label="power in MW", font_size=25, legend=True)
    ax[1].plot(rpms, np.asarray(losses)*3/1e3, lw=3)
    helper.handle_axis(ax[1], title="Power loss", grid=True, x_label="n in rpm", y_label="power loss in kW",
                       font_size=25)
    plt.close(helper.handle_figure(fig, "q2.png", size=(15,10)))


if do["3"] or do["4"]:
    V_grid = 33e3/np.sqrt(3)
    frequency_grid = 2*np.pi*50
    transformer_ratio = 4/33
    cable_length = 65
    R_cable = 0.1*cable_length
    inductance_cable = 0.5e-3*cable_length
    capacitance_cable = 0.1e-6*cable_length

    R_cable_ref_prim, ind_cable_ref_prim, cap_cable_ref_prim = calculate.refer_impedances_to_primary(transformer_ratio,
                                                                                                     R_cable,
                                                                                                     inductance_cable,
                                                                                                     capacitance_cable)
    V_grid_ref_prim, = calculate.refer_voltages_to_primary(transformer_ratio, V_grid)
    imp_sec_ref_prim = 25e-3+R_cable_ref_prim+1j*(0.6e-3+ind_cable_ref_prim)
    if do["3"]:
        I_poc = list()
        rpms = np.arange(2,10)
        power_vals = [90748.5288663250,306276.284923847,725988.230930600,1417945.76353633,
                      2450210.27939077,3890843.17514368,5807905.84744480,8269459.69294386]
        for rpm, mech_power in zip(rpms, power_vals):
            angular_freq = np.pi*rpm*180/30
            I_g, V_g, P_loss, _ = calculate.gen_output(P_mech=mech_power, generator_angular_frequency=angular_freq,
                                                    magnetic_flux=19.49, resistance=64e-3, reactance=1.8e-3)
            power = I_g*V_g
            I_poc_ref_prim, I_1, V_B, I_t, I_2_ref_prim = calculate.calculations(input_power=power*(1+0.2j),
                                                                       V_grid_ref_prim= V_grid_ref_prim,
                                                                       grid_frequency= frequency_grid,
                                                                       core_inductance= 0.046,
                                                                       cable_capacity_ref_prim= cap_cable_ref_prim,
                                                                       impedance_primary= 25e-3+1j*0.6e-3,
                                                                       impedance_sec_ref_prim= imp_sec_ref_prim)
            print(I_1, I_2_ref_prim, I_poc_ref_prim, I_t,V_g)
            I_poc.append(calculate.refer_currents_to_primary(1/transformer_ratio, I_poc_ref_prim[0])[0])
        fig, ax = plt.subplots()
        efficiency_3 = [p_out/p_in for p_in, p_out in zip(power_vals, [3*I.real*V_grid for I in I_poc])]
        active, reactive = [3*I.real*V_grid/1e6 for I in I_poc], [3*I.imag*V_grid/1e6 for I in I_poc]
        df = pd.DataFrame({"n in rpm": rpms, "active power in MW": active, "reactive power in MW": reactive})
        df.to_latex("tab_3", index=False)
        ax.plot(rpms, [3*I.real*V_grid/1e6 for I in I_poc], label="Active power", lw=3)
        ax.plot(rpms, [3*I.imag*V_grid/1e6 for I in I_poc], label="Reactive power", lw=3)
        helper.handle_axis(ax, "Active and reactive power at POC",
                           grid=True, legend=True, x_label="n in rpm", y_label="power in MW", font_size=30)
        plt.close(helper.handle_figure(fig, "q3.png"))


    if do["4"]:
        I_poc = list()
        rpms = np.arange(2,10)
        power_vals = [90748.5288663250,306276.284923847,725988.230930600,1417945.76353633,
                      2450210.27939077,3890843.17514368,5807905.84744480,8269459.69294386]
        rpfs = list()
        for rpm, mech_power in zip(rpms, power_vals):
            angular_freq = np.pi*rpm*180/30
            I_g, V_g, P_loss = calculate.gen_output(mech_power, angular_freq, 19.49, 64e-3, 1.8e-3)
            power = I_g*V_g
            def residue(reactive_power_fac: float) -> float:
                I_poc_ref_prim, _ = calculate.calculations(input_power=power*(1+1j*reactive_power_fac),
                                                           V_grid_ref_prim=V_grid_ref_prim, grid_frequency=frequency_grid,
                                                           core_inductance=0.046,
                                                           cable_capacity_ref_prim=cap_cable_ref_prim,
                                                           impedance_primary=25e-3+1j*0.6e-3,
                                                           impedance_sec_ref_prim=imp_sec_ref_prim)[0]
                return I_poc_ref_prim.imag
            RPF = newton(residue, x0=-0.5)
            rpfs.append(RPF)
            I_poc_ref_prim, I_1, V_B, I_t, I_2_ref_prim = calculate.calculations(input_power=power*(1+1j*RPF),
                                                                                 V_grid_ref_prim= V_grid_ref_prim,
                                                                                 grid_frequency= frequency_grid,
                                                                                 core_inductance= 0.046,
                                                                                 cable_capacity_ref_prim= cap_cable_ref_prim,
                                                                                 impedance_primary= 25e-3+1j*0.6e-3,
                                                                                 impedance_sec_ref_prim= imp_sec_ref_prim)
            I_poc.append(calculate.refer_currents_to_primary(1/transformer_ratio, I_poc_ref_prim[0])[0])
        fig, ax = plt.subplots()
        ax.plot(rpms, [3*I.real*V_grid/1e6 for I in I_poc], label="Active power", lw=3)
        ax.plot(rpms, [3*I.imag*V_grid/1e6 for I in I_poc], label="Reactive power", lw=3)
        active, reactive = [3*I.real*V_grid/1e6 for I in I_poc], [3*I.imag*V_grid/1e6 for I in I_poc]
        df = pd.DataFrame({"n in rpm": rpms, "active power in MW": active, "reactive power in MW": reactive,
                           "RPF": np.asarray(rpfs)})
        df.to_latex("tab_4", index=False)
        helper.handle_axis(ax, "Active and reactive power at POC",
                           grid=True, legend=True, x_label="n in rpm", y_label="power in MW", font_size=30)
        plt.close(helper.handle_figure(fig, "q4.png"))
        fig, ax = plt.subplots()
        ax.plot(rpms, rpfs, lw=3)
        helper.handle_axis(ax, "Reactive power factor", grid=True, x_label="n in rpm", y_label="RPF", font_size=30)
        plt.close(helper.handle_figure(fig, "q4_rpf.png"))

        if do["5"]:
            efficiency = [p_out/p_in for p_in, p_out in zip(power_vals, [3*I.real*V_grid for I in I_poc])]
            fig, ax = plt.subplots()
            ax.plot(rpms, efficiency, lw=5, label="controlled VSC-B")
            ax.plot(rpms, efficiency_3, lw=5, label="Constant VSC-B settings")
            helper.handle_axis(ax, "Efficiency with a controlled VSC-B", grid=True, x_label="n in rpm",
                               y_label="efficiency", font_size=30, legend=True)
            ax.ticklabel_format(useOffset=False)
            plt.close(helper.handle_figure(fig, "q5.png"))
            df = pd.DataFrame({"n in rpm": rpms, "constant VSC-B setting": efficiency_3, "controlled VSC-B": efficiency})
            df.to_latex("tab_5", index=False)
