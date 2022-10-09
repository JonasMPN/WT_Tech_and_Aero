import numpy as np
from scipy.optimize import brentq as root, newton
import pandas as pd
from data_handling import AirfoilInterpolator, BemData
from copy import copy

class BEM:
    def __init__(self,
                 airfoil_data_dir: str,
                 dir_save: str,
                 blade_data_file: str,
                 aerodynamic_data_files: str or list,
                 to_interpolate: dict):
        self.interpolator = AirfoilInterpolator(airfoil_data_dir, aerodynamic_data_files).interpolate(to_interpolate)
        self.bem_data = BemData(dir_save)
        self.df_blade_data = pd.read_csv(airfoil_data_dir+"/"+blade_data_file, index_col=None)
        self.v0 = None
        self.rotor_radius = None
        self.n_blades = None
        self.air_density = None
        self.omega = None

    def set_constants(self,
                      rotor_radius: float,
                      n_blades: int,
                      v0: float,
                      air_density: float) -> None:
        self._set(**{param: value for param, value in locals().items() if param != "self"})
        return None

    def solve(self,
              v0: float,
              omega: float,
              pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.v0 = v0
        self.omega = omega
        self._assert_values()
        radii, normal_force, tangential_force = list(), list(), list()
        for _, data in self.df_blade_data.iterrows():
            radius, chord, twist = data["radius"], data["chord"], data["twist"]
            rel_thickness = data["rel_thickness"]
            phi = self._root_phi(radius, chord, twist, pitch, rel_thickness)
            results = self._phi_to_all(phi, radius, chord, twist, pitch, rel_thickness)
            radii.append(radius)
            normal_force.append(results["normal_force"])
            tangential_force.append(results["tangential_force"])
        return np.asarray(radii), np.asarray(normal_force), np.asarray(tangential_force)

    def optimise(self,
                 tsr_interval: tuple,
                 pitch_interval: tuple,
                 resolution: int=50,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tip_speed_ratios, c_Ps, pitch_angle = list(), list(), list()
        swept_area = np.pi*self.rotor_radius**2
        flow_power = 0.5*self.air_density*swept_area*self.v0**3
        c_Ps = np.zeros((resolution, resolution))
        TSRs, pitch_angles = np.linspace(*tsr_interval, resolution), np.deg2rad(np.linspace(*pitch_interval, resolution))
        for tsr_i, tpr in enumerate(TSRs):
            print(f"Finished {np.round(tsr_i/len(TSRs)*100,3)}%.")
            tip_speed_ratios.append(tpr)
            omega = tpr*self.v0/self.rotor_radius
            for pitch_i, pitch in enumerate(pitch_angles):
                power, _ = self._thrust_and_power(self.v0, omega, pitch)
                c_Ps[pitch_i, tsr_i] = power/flow_power
                pitch_angle.append(pitch)
        pitch_angles, tip_speed_ratios = np.meshgrid(pitch_angles, TSRs)
        self.bem_data.save(resolution=resolution, pitch=pitch_angles, tsr=tip_speed_ratios)
        return tip_speed_ratios, pitch_angles, c_Ps

    def pitch_curve(self,
                    rated_power: float,
                    wind_speeds: tuple,
                    pitch_step_size: float,
                    optimums_from_res: int=None,
                    tsr_optimum: float=None,
                    pitch_optimum: float=None,
                    loop_breaker: int=2000) -> None:
        """

        :param rated_power:
        :param wind_speeds: (v0_min, v0_max, resolution)
        :param resolution:
        :param pitch_step_size: in degree
        :param optimums_from_res:
        :param tsr_optimum:
        :param pitch_optimum: has to be a float
        :param loop_breaker:
        :return:
        """
        pitch_step_size, pitch_optimum = np.deg2rad(pitch_step_size), float(pitch_optimum)
        init_pitch_step_size = pitch_step_size
        if optimums_from_res:
            tsr_optimum, pitch_optimum = self.bem_data.optimum_from_resolution(optimums_from_res)
        power, pitch, pitch_steps, pitch_step_size, plot_pss, v0s = dict(), dict(), dict(), dict(), dict(), dict()
        c_P, c_T = dict(), dict()
        for key in ["ramp", "feather", "stall"]:
            power[key], pitch[key], pitch_steps[key], pitch_step_size[key] = list(), list(), list(), list()
            v0s[key], plot_pss[key], c_P[key], c_T[key] = list(), list(), list(), list()
        pitch["feather"], pitch["stall"] = [pitch_optimum], [pitch_optimum]
        pitch_step_size["feather"], pitch_step_size["stall"] = [copy(init_pitch_step_size)], [copy(init_pitch_step_size)]
        omega, v_rated = 0, 0
        for i, v0 in enumerate(np.linspace(*wind_speeds)):
            power_present = .5*self.air_density*v0**3*np.pi*self.rotor_radius**2
            thrust_present = .5*self.air_density*v0**2*np.pi*self.rotor_radius**2
            omega = tsr_optimum * v0 / self.rotor_radius
            power_now, thrust = self._thrust_and_power(v0, omega, pitch_optimum)
            v_rated = v0
            if power_now > rated_power:
                break
            power["ramp"].append(power_now)
            pitch["ramp"].append(pitch_optimum)
            v0s["ramp"].append(v0)
            c_P["ramp"].append(power_now/power_present)
            c_T["ramp"].append(thrust/thrust_present)
            print(f"Finished {np.round(i / wind_speeds[2] * 100, 5)}%.")
        factor = {"feather":
                      {"decrease": 1,
                       "increase": -1},
                  "stall":
                      {"decrease": -1,
                       "increase": 1}}
        compare = {"decrease": max, "increase": min}
        n_v0_ramp = len(v0s["ramp"])
        for i, v0 in enumerate(np.linspace(v_rated, wind_speeds[1], wind_speeds[2]-n_v0_ramp)):
            power_present = .5*self.air_density*v0**3*np.pi*self.rotor_radius**2
            thrust_present = .5*self.air_density*v0**2*np.pi*self.rotor_radius**2
            for control_type in ["feather", "stall"]:
                pitch[control_type].append(pitch[control_type][-1])
                v0s[control_type].append(v0)
                power_now, thrust = self._thrust_and_power(v0, omega, pitch[control_type][-1])
                to_do = "decrease" if power_now > rated_power else "increase"
                pitch_change = factor[control_type][to_do]*pitch_step_size[control_type][-1]
                counter = 0
                while compare[to_do](power_now, rated_power) == power_now:
                    pitch[control_type][-1] += pitch_change
                    power_now, thrust = self._thrust_and_power(v0, omega, pitch[control_type][-1])
                    counter += 1
                    if counter > loop_breaker:
                        self._raise_loop_break_error(control_type, counter, pitch_step_size[control_type][-1])
                pitch_steps[control_type].append(counter)
                power[control_type].append(power_now)
                c_P[control_type].append(power_now/power_present)
                c_T[control_type].append(thrust/thrust_present)
                if counter > 15:
                    pitch_step_size[control_type].append(pitch_step_size[control_type][-1]*1.3)
                elif counter < 10:
                    pitch_step_size[control_type].append(pitch_step_size[control_type][-1]/1.3)
                else:
                    pitch_step_size[control_type].append(pitch_step_size[control_type][-1])
                plot_pss[control_type].append(pitch_change)
            print(f"Finished {np.round((n_v0_ramp+i)/wind_speeds[2]*100, 5)}%.")
        resolution = np.rad2deg(init_pitch_step_size)
        if int(resolution) == resolution:
            resolution = int(resolution)
        pitch["feather"].pop(0), pitch["stall"].pop(0)
        self.bem_data.save_dataframe(add_to_filename="_ramp", resolution=resolution,
                                     v0=v0s["ramp"], power=power["ramp"], pitch=pitch["ramp"], cT=c_T["ramp"],
                                     cP=c_P["ramp"])
        self.bem_data.save_dataframe(add_to_filename="_control", resolution=resolution,
                                     v0=v0s["stall"],
                                     pitch_feather=pitch["feather"], power_feather=power["feather"],
                                     pitch_stall=pitch["stall"], power_stall=power["stall"],
                                     feather_pitch_steps=pitch_steps["feather"], stall_pitch_steps=pitch_steps["stall"],
                                     feather_step_size=plot_pss["feather"], stall_step_size=plot_pss["stall"],
                                     feather_cP=c_P["feather"], stall_cP=c_P["stall"],
                                     feather_cT=c_T["feather"], stall_cT=c_T["stall"])

    def _thrust_and_power(self, v0, omega, pitch) -> tuple[float, float]:
        radii, _, F_t = self.solve(v0, omega, pitch)
        thrust = self.n_blades * np.trapz(F_t, radii)
        torque = self.n_blades * np.trapz(F_t*radii, radii)
        return torque*omega, thrust

    def _root_phi(self,
                  radius: float,
                  chord: float,
                  twist: float,
                  pitch: float,
                  rel_thickness: float,
                  bracket=(1e-5, np.pi/2)) -> float:
        def residue(phi):
            alpha = phi-(pitch+twist)
            c_lift = self.interpolator["c_l"](rel_thickness, alpha)
            c_drag = self.interpolator["c_d"](rel_thickness, alpha)
            if np.isnan(c_lift) or np.isnan(c_drag):
              raise ValueError(f"Trying to extrapolate data for relative thickness {rel_thickness}, alpha {alpha}. "
                               f"Only interpolation allowed.")
            c_normal = self._c_normal(phi, c_lift, c_drag)
            c_tangent = self._c_tangent(phi,  c_lift, c_drag)
            tip_loss_correction = self._tip_loss_correction(radius, phi, self.rotor_radius, self.n_blades)
            local_solidity = self._local_solidity(chord, radius, self.n_blades)
            axial_induction_factor = self._root_axial_induction_factor(phi, local_solidity, c_normal, tip_loss_correction)
            tangential_induction_factor = self._tangential_induction_factor(phi, local_solidity, c_tangent, tip_loss_correction)
            lhs = np.sin(phi)/(1-axial_induction_factor)
            rhs = self.v0*np.cos(phi)/(self.omega*radius*(1+tangential_induction_factor))
            return lhs-rhs
        # print("phi:", residue(bracket[0]), residue(bracket[1]))
        return root(residue, *bracket)
        # return newton(residue, .1)

    def _phi_to_all(self,
                    phi: float,
                    radius: float,
                    chord: float,
                    twist: float,
                    pitch: float,
                    rel_thickness: float) -> dict:
        alpha = phi-(pitch+twist)
        c_lift = self.interpolator["c_l"](rel_thickness, alpha)
        c_drag = self.interpolator["c_d"](rel_thickness, alpha)
        local_solidity = self._local_solidity(chord, radius, self.n_blades)
        c_normal = self._c_normal(phi, c_lift, c_drag)
        c_tangent = self._c_tangent(phi,  c_lift, c_drag)
        tip_loss_correction = self._tip_loss_correction(radius, phi, self.rotor_radius, self.n_blades)
        axial_if = self._root_axial_induction_factor(phi, local_solidity, c_normal, tip_loss_correction)
        tangential_if = self._tangential_induction_factor(phi, local_solidity, c_tangent, tip_loss_correction)
        relative_flow = np.sqrt((self.omega*radius*(1+tangential_if))**2+(self.v0*(1-axial_if))**2)
        normal_force = 0.5*self.air_density*relative_flow**2*chord*c_normal
        tangential_force = 0.5*self.air_density*relative_flow**2*chord*c_tangent
        return {
            "phi": phi,
            "axial_if": axial_if,
            "tangential_if": tangential_if,
            "normal_force": normal_force,
            "tangential_force": tangential_force,
            "tip_loss_correction": tip_loss_correction,
        }

    @staticmethod
    def _root_axial_induction_factor(phi: float,
                                     local_solidity: float,
                                     c_normal: float,
                                     tip_loss_correction: float,
                                     bracket: tuple=(-1,1)) -> float:
        def residue(aif):
            if aif <= 1/3:
                return 1/((4*tip_loss_correction*np.sin(phi)**2)/(local_solidity*c_normal)+1)-aif
            else:
                return local_solidity*((1-aif)/np.sin(phi))**2*c_normal-4*aif*tip_loss_correction*(1-aif/4*(5-3*aif))
        # print(phi, residue(bracket[0]), residue(bracket[1]))
        try:
            return root(residue, *bracket)
        except ValueError:
            print(c_normal, tip_loss_correction)
            print()
        # return newton(residue, 0)

    def _set(self, **kwargs) -> None:
        """
        Sets parameters of the instance. Raises an error if a parameter is trying to be set that doesn't exist.
        :param kwargs:
        :return:
        """
        params = self.__dict__
        existing_parameters = [*params]
        for parameter, value in kwargs.items():
            if parameter not in existing_parameters:
                raise ValueError(f"Parameter {parameter} cannot be set. Settable parameters are {existing_parameters}.")
            params[parameter] = value
        return None

    def _assert_values(self):
        not_set = list()
        for variable, value in vars(self).items():
            if value is None:
                not_set.append(variable)
        if len(not_set) != 0:
            raise ValueError(f"Variable(s) {not_set} not set. Set all variables before use.")

    @staticmethod
    def _c_normal(phi: float, c_lift: float, c_drag: float) -> float:
        """
        Calculates an aerodynamic "lift" coefficient according to a coordinate transformation with phi
        :param phi: angle between flow and rotational direction in rad
        :param c_lift: lift coefficient old coordinate system
        :param c_drag: lift coefficient old coordinate system
        :return: Normal force in Newton
        """
        return c_lift*np.cos(phi)+c_drag*np.sin(phi)

    @staticmethod
    def _c_tangent(phi: float, c_lift: float, c_drag: float) -> float:
        """
        Calculates an aerodynamic "drag" coefficient according to a coordinate transformation with phi
        :param phi: angle between flow and rotational direction in rad
        :param c_lift: lift coefficient old coordinate system
        :param c_drag: lift coefficient old coordinate system
        :return: Normal force in Newton
        """
        return c_lift*np.sin(phi)-c_drag*np.cos(phi)

    @staticmethod
    def _local_solidity(chord: float, radius: float, n_blades: int) -> float:
        """
        Calculates the local solidity
        :param chord: in m
        :param radius: distance from rotor axis to blade element in m
        :param n_blades: number of blades
        :return: local solidity
        """
        return n_blades*chord/(2*np.pi*radius)

    @staticmethod
    def _tip_loss_correction(r: float, phi: float, rotor_radius: float, n_blades: int) -> float:
        """
        Returns the factor F for the tip loss correction according to Prandtl
        :param r: current radius
        :param phi: angle between flow and rotational direction in rad
        :param rotor_radius: total radius of the rotor
        :param n_blades: number of blades
        :return: Prandtl tip loss correction
        """
        if np.sin(np.abs(phi)) < 0.01:
            return 1
        return 2/np.pi*np.arccos(np.exp(-(n_blades*(rotor_radius-r))/(2*r*np.sin(np.abs(phi)))))

    @staticmethod
    def _tangential_induction_factor(phi: float, local_solidity: float, c_tangent: float, tip_loss_correction: float)\
            -> float:
        return 1/((4*tip_loss_correction*np.sin(phi)*np.cos(phi))/(local_solidity*c_tangent)-1)

    @staticmethod
    def _raise_loop_break_error(control_type: str, counter: int, step_size: float):
        raise ValueError(f"Could not pitch to rated power with {counter*step_size}°change {control_type}. The number "
                         f"of pitch increments (loop_breaker) might have been too low.")

