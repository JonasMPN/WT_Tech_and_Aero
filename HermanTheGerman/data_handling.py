import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from helper_functions import Helper
import ast
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

        tsr = np.load(self.dir_data+f"/tsr_{resolution}.npy")
        pitch = np.load(self.dir_data+f"/pitch_{resolution}.npy")
        cps = np.load(self.dir_data+f"/cps_{resolution}.npy")
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(tsr, pitch, cps, **surface_kwargs)
        helper.handle_axis(ax, **axes_kwargs)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/contour_{resolution}{add_to_fig_name}.png", **figure_kwargs))
        return None

    def pitch_curve(self,
                    pitch_step_size: float,
                    axes_kwargs: dict=None,
                    figure_kwargs: dict=None,
                    add_to_fig_name: str="") -> None:
        """

        :param pitch_step_size: degree
        :param axes_kwargs:
        :param figure_kwargs:
        :param add_to_fig_name:
        :return:
        """
        axes_kwargs = {} if axes_kwargs is None else axes_kwargs
        figure_kwargs = {} if figure_kwargs is None else figure_kwargs

        df_ramp = pd.read_csv(self.dir_data+f"/{pitch_step_size}_ramp.dat", index_col=None)
        df_control = pd.read_csv(self.dir_data+f"/{pitch_step_size}_control.dat", index_col=None)

        fig, ax = plt.subplots()
        ax.plot(df_ramp["v0"], df_ramp["power"], "k", label="ramp up")
        ax.plot(df_control["v0"], df_control["power_stall"], label="stall")
        ax.plot(df_control["v0"], df_control["power_feather"], label="feather")
        helper.handle_axis(ax, title="Power curve", legend=True, x_label=r"$v_0$ in $\frac{m}{s}$", y_label="power in W",
                           font_size=20)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/power_{pitch_step_size}.png"))

        fig, ax = plt.subplots()
        ax.plot(df_ramp["v0"], df_ramp["pitch"], "k", label="ramp up")
        ax.plot(df_control["v0"], df_control["pitch_stall"].apply(np.rad2deg), label="stall")
        ax.plot(df_control["v0"], df_control["pitch_feather"].apply(np.rad2deg), label="feather")
        helper.handle_axis(ax, title="Pitch curve", legend=True, x_label=r"$v_0$ in $\frac{m}{s}$", y_label="pitch in °",
                           font_size=20)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/pitch_{pitch_step_size}.png"))

        fig, ax = plt.subplots()
        ax.plot(df_control["v0"], df_control["stall_pitch_steps"], label="stall")
        ax.plot(df_control["v0"], df_control["feather_pitch_steps"], label="feather")
        helper.handle_axis(ax, title="Pitch steps", legend=True, x_label=r"$v_0$ in $\frac{m}{s}$",
                           y_label="pitch steps", font_size=20)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/pitch_steps_{pitch_step_size}.png"))

        fig, ax = plt.subplots()
        ax.plot(df_control["v0"], df_control["stall_step_size"], label="stall")
        ax.plot(df_control["v0"], df_control["feather_step_size"], label="feather")
        helper.handle_axis(ax, title="Pitch step size", legend=True, x_label=r"$v_0$ in $\frac{m}{s}$",
                           y_label="pitch step size", font_size=20)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/pitch_step_size_{pitch_step_size}.png"))

        fig, ax = plt.subplots()
        ax.plot(df_ramp["v0"], df_ramp["cP"], label="ramp")
        ax.plot(df_control["v0"], df_control["feather_cP"], label="feather")
        ax.plot(df_control["v0"], df_control["stall_cP"], label="stall")
        helper.handle_axis(ax, title="Power coefficient", legend=True, x_label=r"$v_0$ in $\frac{m}{s}$",
                           y_label=r"$c_P$", font_size=20)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/cP_{pitch_step_size}.png"))

        fig, ax = plt.subplots()
        ax.plot(df_ramp["v0"], df_ramp["cT"], label="ramp")
        ax.plot(df_control["v0"], df_control["feather_cT"], label="feather")
        ax.plot(df_control["v0"], df_control["stall_cT"], label="stall")
        helper.handle_axis(ax, title="Thrust coefficient", legend=True, x_label=r"$v_0$ in $\frac{m}{s}$",
                           y_label=r"$c_t$", font_size=20)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/cT_{pitch_step_size}.png"))
        return None

    def optimum_from_resolution(self, resolution: int) -> tuple[float, float, float]:
        tsr = np.load(self.dir_data+f"/tip_speed_ratios_{resolution}.npy")
        pitch = np.load(self.dir_data+f"/pitch_angles_{resolution}.npy")
        cps = np.load(self.dir_data+f"/c_Ps_{resolution}.npy")
        id_max = np.unravel_index(cps.argmax(), (resolution, resolution))
        return np.max(cps), tsr[id_max], pitch[id_max]

    def save_dataframe(self, add_to_filename:str="", **kwargs):
        """
        :param kwargs: One kwarg has to be 'resolution'
        :return:
        """
        df = pd.DataFrame()
        resolution = kwargs["resolution"]
        kwargs.pop("resolution")
        for parameter, values in kwargs.items():
            df[parameter] = values
        df.to_csv(self.dir_data+f"/{resolution}{add_to_filename}.dat", index=False)

    def save(self, **kwargs) -> None:
        """
        :param kwargs: One kwarg has to be 'resolution'
        :return:
        """
        resolution = kwargs["resolution"]
        kwargs.pop("resolution")
        for parameter, values in kwargs.items():
            np.save(self.dir_data+f"/{parameter}_{resolution}.npy", values)
        return None


class AshesData:
    def __init__(self, data_dir: str):
        self.dir_data = data_dir

    def ascii_to_dat(self, file: str, file_extension: str="dat") -> None:
        columns = list()
        be_positions = list()
        with open(self.dir_data+f"/{file}") as data_file:
            content = data_file.readlines()
            columns_str = content[11]
            content_cleaned = False
            while not content_cleaned:
                columns.append(columns_str[:columns_str.find("[") - 1])
                columns_str = columns_str[columns_str.find("\t") + 1:]
                if "\t" not in columns_str:
                    content_cleaned = True
            position_str = content[20]
            got_all = False
            while not got_all:
                be_positions.append(float(position_str[:position_str.find(" ")]))
                position_str = position_str[position_str.find(" ") + 1:]
                if " " not in position_str:
                    got_all = True
            n_blade_elements = len(be_positions)
            data = {column: list() for column in ["radius"]+columns}
            for time_data in content[22:]:
                time = float(time_data[:time_data.find("\t")])
                time_data = time_data[time_data.find("\t")+1:]
                data["Time"] += [time for _ in range(n_blade_elements)]
                data["radius"] += be_positions
                for col in columns[1:]:
                    values = list(ast.literal_eval(time_data[:time_data.find("\t")-1].replace(" ", ",")))
                    data[col] += values
            df = pd.DataFrame(data)
            file_save = file[:file.find('.')]+f".{file_extension}"
            df.to_csv(self.dir_data+f"/{file_save}", index=False)