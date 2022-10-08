import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from helper_functions import Helper
helper = Helper()


class AirfoilInterpolator:
    def __init__(self,
                 dir_data: str,
                 data_profiles: list or str,
                 **pandas_read_csv):
        if type(data_profiles) == str:
            data_profiles = [data_profiles]
        self.dir_data = dir_data
        self.df_profiles = list()
        for file in data_profiles:
            self.df_profiles.append(pd.read_csv(dir_data+"/"+file, **pandas_read_csv))
    
    def interpolate(self,
                    to_interpolate: dict) -> dict:
        """
        Multidimensional interpolation. The data is taken from the files that are specified during the initialisation of
        the class' instance. The parameter 'to_interpolate' states which parameter(s) is(are) to be interpolated and on
        which arguments this interpolation is based. The keys of 'to_interpolate' specify the parameters that will be
        interpolated. The values of each key specify on which arguments the interpolation is based upon.
        :param to_interpolate: keys: values to interpolate, values: arguments for interpolation
        :return:
        """
        interpolator = {var: None for var in to_interpolate}
        for to_inter, arguments in to_interpolate.items():
            if type(arguments) == str:
                arguments = [arguments]
            points = {var: list() for var in arguments}
            values = list()
            for df in self.df_profiles:
                for _, row in df.iterrows():
                    for arg in arguments:
                        points[arg].append(row[arg])
                    values.append(row[to_inter])
            points = np.asarray([all_points for all_points in points.values()]).T
            interpolator[to_inter] = interpolate.LinearNDInterpolator(points, values)
        return interpolator


class BemData:
    def __init__(self, data_dir: str):
        self.dir_data = data_dir

    def save(self,
             resolution: int,
             tip_speed_ratios: np.ndarray,
             pitch_angles: np.ndarray,
             c_Ps: np.ndarray) -> None:
        np.save(f"data/tsr_{resolution}.npy", tip_speed_ratios)
        np.save(f"data/pitch_{resolution}.npy", pitch_angles)
        np.save(f"data/cps_{resolution}.npy", c_Ps)
        return None

    def contourf(self,
                 resolution: int,
                 contourf_kwargs: dict=None,
                 axes_kwargs: dict=None,
                 figure_kwargs: dict=None,
                 add_to_fig_name: str="") -> None:
        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        axes_kwargs = {} if axes_kwargs is None else axes_kwargs
        figure_kwargs = {} if figure_kwargs is None else figure_kwargs

        tsr = np.load(self.dir_data+f"/tsr_{resolution}.npy")
        pitch = np.load(self.dir_data+f"/pitch_{resolution}.npy")
        cps = np.load(self.dir_data+f"/cps_{resolution}.npy")
        fig, ax = plt.subplots()
        CS = ax.contour(tsr, pitch, cps, **contourf_kwargs)
        ax.clabel(CS, inline=True, fontsize=10)
        helper.handle_axis(ax, **axes_kwargs)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/contour_{resolution}{add_to_fig_name}.png", **figure_kwargs))
        return None

    def surface(self,
                resolution: int,
                surface_kwargs: dict=None,
                axes_kwargs: dict=None,
                figure_kwargs: dict=None,
                add_to_fig_name: str="") -> None:
        surface_kwargs = {} if surface_kwargs is None else surface_kwargs
        axes_kwargs = {} if axes_kwargs is None else axes_kwargs
        figure_kwargs = {} if figure_kwargs is None else figure_kwargs

        tsr = np.load(self.dir_data+f"/tsr_{resolution}/.n[y")
        pitch = np.load(self.dir_data+f"/pitch_{resolution}/.n[y")
        cps = np.load(self.dir_data+f"/cps_{resolution}/.n[y")
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(tsr, pitch, cps, **surface_kwargs)
        helper.handle_axis(ax, **axes_kwargs)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/contour_{resolution}{add_to_fig_name}.png", **figure_kwargs))
        return None