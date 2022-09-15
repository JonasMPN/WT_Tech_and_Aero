import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq as root

class BEM():
    def __init__(self):
        self.last_axial_induction_factor = 0

        self.v0 = None
        self.omega = None
        self.rotor_radius = None
        self.n_blades = None
        self.air_density = None
        self.twist = None
        self.pitch = None

    def set_constants(self, v0: float, omega: float, rotor_radius: float, n_blades: int, air_density: float) -> None:
        self.v0, self.omega, self.rotor_radius, self.n_blades = v0, omega, rotor_radius, n_blades
        self.air_density = air_density
        return None

    def solve(self, radius, chord, c_lift, c_drag) -> dict:
        # if not all([self.v0, self.omega, self.rotor_radius, self.n_blades, self.air_density, self.twist, self.pitch]):
        #     print(f"\033[38;2;{255};{0};{0}mAll constants must be set first. \033[38;2;255;255;255m")
        #     return False
        # for r in np.linspace(0.1, self.rotor_radius, n_blade_elements):
        #     pass
        self._reset_values()
        phi = self._root_phi(radius, chord, c_lift, c_drag)
        return self._phi_to_all(phi, radius, chord, c_lift, c_drag)

    def _phi_to_all(self, phi: float, radius: float, chord: float, c_lift: float, c_drag: float) -> dict:
        local_solidity = self._local_solidity(chord, radius, self.n_blades)
        c_normal = self._c_normal(phi, c_lift, c_drag)
        c_tangent = self._c_tangent(phi,  c_lift, c_drag)
        tip_loss_correction = self._tip_loss_correction(radius, phi, self.rotor_radius, self.n_blades)
        axial_if = self._axial_induction_factor(phi, local_solidity, c_normal, tip_loss_correction)
        tangential_if = self._tangential_induction_factor(phi, local_solidity, c_tangent, tip_loss_correction)
        relative_flow = np.sqrt((self.omega*radius*(1+tangential_if))**2+(self.v0*(1-axial_if))**2)
        normal_force = 0.5*self.air_density*relative_flow**2*chord*c_normal
        tangential_force = 0.5*self.air_density*relative_flow**2*chord*c_tangent
        return {
            "tip_loss_correction": tip_loss_correction,
            "axial_if": axial_if,
            "tangential_if": tangential_if,
            "normal_force": normal_force,
            "tangential_force": tangential_force
        }

    def _axial_induction_factor(self, phi: float, local_solidity: float, c_normal: float, tip_loss_correction: float) \
            -> float:
        if self.last_axial_induction_factor <= 1/3:
            self.last_axial_induction_factor = 1/((4*tip_loss_correction*np.sin(phi)**2)/(local_solidity*c_normal)+1)
        else:
            def to_solve(a):
                return local_solidity*((1-a)/np.sin(phi))**2*c_normal-4*a*tip_loss_correction*(1-a/4*(5-3*a))
            self.last_axial_induction_factor = root(to_solve, 0, 1)
        return self.last_axial_induction_factor

    def _root_phi(self, r: float, chord: float, c_lift: float, c_drag: float, bracket=(-0.1,np.pi/2)) -> tuple:
        def residue(phi):
            c_normal = self._c_normal(phi, c_lift, c_drag)
            c_tangent = self._c_tangent(phi,  c_lift, c_drag)
            tip_loss_correction = self._tip_loss_correction(r, phi, self.rotor_radius, self.n_blades)
            local_solidity = self._local_solidity(chord, r, self.n_blades)
            axial_induction_factor = self._axial_induction_factor(phi, local_solidity, c_normal, tip_loss_correction)
            tangential_induction_factor = self._tangential_induction_factor(phi, local_solidity, c_tangent, tip_loss_correction)
            return np.sin(phi)/(1-axial_induction_factor)-self.v0*np.cos(phi)/(self.omega*r*(1+tangential_induction_factor))
        return root(residue, bracket[0], bracket[1])

    def _reset_values(self, axial_induction_factor: float=0):
        self.last_axial_induction_factor = axial_induction_factor

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
        return 2/np.pi*np.arccos(np.exp(-(n_blades*(rotor_radius-r))/(2*r*np.sin(np.abs(phi)))))

    @staticmethod
    def _tangential_induction_factor(phi: float, local_solidity: float, c_tangent: float, tip_loss_correction: float)\
            -> float:
        return 1/((4*tip_loss_correction*np.sin(phi)*np.cos(phi))/(local_solidity*c_tangent)-1)


