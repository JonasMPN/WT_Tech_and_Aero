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
        cps = np.load(self.dir_data+f"/c_Ps_{resolution}.npy")
        fig, ax = plt.subplots()
        CS = ax.contour(tsr, pitch, cps, **contourf_kwargs)
        ax.clabel(CS, inline=True, fontsize=10)
        helper.handle_axis(ax, **axes_kwargs)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/contour_{resolution}{add_to_fig_name}.png", **figure_kwargs))
        return None

    def surface(self,
                resolution: int,                surface_kwargs: dict=None,
                axes_kwargs: dict=None,
                figure_kwargs: dict=None,
                add_to_fig_name: str="") -> None:
        surface_kwargs = {} if surface_kwargs is None else surface_kwargs
        axes_kwargs = {} if axes_kwargs is None else axes_kwargs
        figure_kwargs = {} if figure_kwargs is None else figure_kwargs

        tsr = np.load(self.dir_data+f"/tsr_{resolution}.npy")
        pitch = np.load(self.dir_data+f"/pitch_{resolution}.npy")
        cps = np.load(self.dir_data+f"/c_Ps_{resolution}.npy")
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(tsr, pitch, cps, **surface_kwargs)
        helper.handle_axis(ax, **axes_kwargs)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/surface_{resolution}{add_to_fig_name}.png",
                                       **figure_kwargs))
        return None

    def pitch_curve(self,
                    init_pitch_step_size: float,
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

        df_ramp = pd.read_csv(self.dir_data+f"/{init_pitch_step_size}_ramp.dat", index_col=None)
        df_control = pd.read_csv(self.dir_data+f"/{init_pitch_step_size}_control.dat", index_col=None)

        v0 = df_ramp["v0"].tolist()+df_control["v0"].tolist()
        df_ramp["power"] = df_ramp["power"]/1e7
        df_ramp["thrust"] = df_ramp["thrust"]/1e6

        df_control["power_feather"] = df_control["power_feather"]/1e7
        df_control["power_stall"] = df_control["power_stall"]/1e7
        df_control["thrust_stall"] = df_control["thrust_stall"]/1e6
        df_control["thrust_feather"] = df_control["thrust_feather"]/1e6


        fig, ax = plt.subplots()
        control_v0 = [df_ramp["v0"].tolist()[-1]] + df_control["v0"].tolist()
        pitch_stall = [df_ramp["pitch"].apply(np.rad2deg).tolist()[-1]]+df_control["pitch_stall"].apply(
            np.rad2deg).tolist()
        feather_stall = [df_ramp["pitch"].apply(np.rad2deg).tolist()[-1]]+df_control["pitch_feather"].apply(
            np.rad2deg).tolist()
        ax.plot(df_ramp["v0"], df_ramp["pitch"].apply(np.rad2deg), "k", label="below rated", lw=3)
        ax.plot(control_v0,pitch_stall, label="stall", lw=3)
        ax.plot(control_v0, feather_stall, label="feather", lw=3)
        helper.handle_axis(ax, title="Pitch curve", legend=True, x_label=r"$v_0$ in $\frac{m}{s}$", y_label="pitch in °",
                           **axes_kwargs)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/pitch_{init_pitch_step_size}.png"))

        fig, ax = plt.subplots()
        ax.plot(v0, (df_ramp["power"].tolist()+df_control["power_stall"].tolist()), "k", label="P", lw=3)
        ax2 = ax.twinx()
        ax2.plot(v0, df_ramp["cP"].tolist()+df_control["stall_cP"].tolist(), label=r"$c_P$", lw=3, color="r")
        helper.handle_axis([ax, ax2], title="Power (stall control)", legend=True,
                           x_label=r"$v_0$ in $\frac{m}{s}$", y_label=["P in 10MW", r"$c_P$"], legend_loc=6,
                           **axes_kwargs)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/power_stall_{init_pitch_step_size}.png"))

        fig, ax = plt.subplots()
        ax.plot(v0, (df_ramp["power"].tolist()+df_control["power_feather"].tolist()), "k", label="P", lw=3)
        ax2 = ax.twinx()
        ax2.plot(v0, df_ramp["cP"].tolist()+df_control["feather_cP"].tolist(), label=r"$c_P$", lw=3, color="r")
        helper.handle_axis([ax, ax2], title="Power (feather control)", legend=True,
                           x_label=r"$v_0$ in $\frac{m}{s}$", y_label=["P in 10MW", r"$c_P$"], legend_loc=6,
                           **axes_kwargs)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/power_feather_{init_pitch_step_size}.png"))

        fig, ax = plt.subplots()
        ax.plot(v0, (df_ramp["thrust"].tolist()+df_control["thrust_feather"].tolist()), "k", label="T", lw=3)
        ax2 = ax.twinx()
        ax2.plot(v0, df_ramp["cT"].tolist()+df_control["feather_cT"].tolist(), label=r"$c_T$", lw=3, color="b")
        helper.handle_axis([ax, ax2], title="Thrust (feather control)", legend=True,
                           x_label=r"$v_0$ in $\frac{m}{s}$", y_label=["T in MN", r"$c_P$"], legend_loc=6,
                           **axes_kwargs)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/thrust_feather_{init_pitch_step_size}.png"))

        fig, ax = plt.subplots()
        ax.plot(v0, (df_ramp["thrust"].tolist()+df_control["thrust_stall"].tolist()), "k", label="T", lw=3)
        ax2 = ax.twinx()
        ax2.plot(v0, df_ramp["cT"].tolist()+df_control["stall_cT"].tolist(), label=r"$c_T$", lw=3, color="b")
        helper.handle_axis([ax, ax2], title="Thrust (stall control)", legend=True, legend_loc=6,
                           x_label=r"$v_0$ in $\frac{m}{s}$", y_label=["T in MN", r"$c_P$"],
                           **axes_kwargs)
        plt.close(helper.handle_figure(fig, self.dir_data + f"/thrust_stall_{init_pitch_step_size}.png"))

        fig, ax = plt.subplots()
        ax.plot(df_control["v0"], df_control["stall_pitch_steps"], label="stall", lw=3)
        ax.plot(df_control["v0"], df_control["feather_pitch_steps"], label="feather", lw=3)
        helper.handle_axis(ax, title="Pitch steps", legend=True, x_label=r"$v_0$ in $\frac{m}{s}$",
                           y_label="pitch steps", **axes_kwargs)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/pitch_steps_{init_pitch_step_size}.png"))

        fig, ax = plt.subplots()
        ax.plot(df_control["v0"], df_control["stall_step_size"], label="stall", lw=3)
        ax.plot(df_control["v0"], df_control["feather_step_size"], label="feather", lw=3)
        helper.handle_axis(ax, title="Pitch step size", legend=True, x_label=r"$v_0$ in $\frac{m}{s}$",
                           y_label="pitch step size", **axes_kwargs)
        plt.close(helper.handle_figure(fig, self.dir_data+f"/pitch_step_size_{init_pitch_step_size}.png"))


    def optimum_from_resolution(self, resolution: int) -> tuple[float, float, float]:
        tsr = np.load(self.dir_data+f"/tip_speed_ratios_{resolution}.npy")
        pitch = np.load(self.dir_data+f"/pitch_angles_{resolution}.npy")
        cps = np.load(self.dir_data+f"/c_Ps__{resolution}.npy")
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

    def ascii_to_dat(self, file: str, as_type: str="dat", save: bool=True) -> None or pd.DataFrame:
        """
        Currently only works for these sensors:
            - Blade [Time]
            - Blade [Span]
            - Rotor
        :param file:
        :param as_type:
        :param save:
        :return:
        """
        implemented_sensors = ["Rotor", "Blade [Span]", "Blade [Time]"]
        information_cols ={
            "Rotor": {"columns": 11, "data_start": 18},
            "Blade [Span]": {"columns": 11, "data_start": 22, "be_positions": 20},
            "Blade [Time]": {"columns": 12, "data_start": 19}
        }
        columns = list()
        be_positions = list()
        with open(self.dir_data+f"/{file}") as data_file:
            content = data_file.readlines()
            sensor = None
            for sens in implemented_sensors:
                if sens in content[0]:
                    sensor = sens
                    print(f"Sensor detected as {sensor}")
                    break
            if sensor is None:
                raise ValueError(f"Data from {content[0]} not supported. Supported sensors are for "
                                 f"{implemented_sensors}.")
            columns_str = content[information_cols[sensor]["columns"]]
            content_cleaned = False
            while not content_cleaned:
                idx_bracket = columns_str.find("[")-1
                try:
                    idx_comma = columns_str.index(",")
                except ValueError:
                    idx_comma = idx_bracket
                idx = idx_bracket if idx_bracket < idx_comma else idx_comma
                columns.append(columns_str[:idx])
                columns_str = columns_str[columns_str.find("\t") + 1:]
                if "\t" not in columns_str:
                    content_cleaned = True
            spanwise = False
            if "Blade" in sensor:
                spanwise = True
                position_str = content[information_cols[sensor]["be_positions"]]
                got_all = False
                while not got_all:
                    be_positions.append(float(position_str[:position_str.find(" ")]))
                    position_str = position_str[position_str.find(" ") + 1:]
                    if " " not in position_str:
                        got_all = True
                n_blade_elements = len(be_positions)
            columns = columns if "Blade" not in sensor else ["radius"]+columns
            data = {column: list() for column in columns}
            for time_data in content[information_cols[sensor]["data_start"]:]:
                if spanwise:
                    time = float(time_data[:time_data.find("\t")])
                    time_data = time_data[time_data.find("\t")+1:]
                    data["Time"] += [time for _ in range(n_blade_elements)]
                    data["radius"] += be_positions
                    for col in columns[2:]:
                        sensor_separation = time_data.find("\t")
                        values = list(ast.literal_eval(time_data[:sensor_separation-1].replace(" ", ",")))
                        for i, char in enumerate(time_data[sensor_separation+1:]):
                            if char not in [" ", "\t"]:
                                time_data = time_data[sensor_separation+1+i:]
                                break
                        data[col] += values
                else:
                    time_data = list(ast.literal_eval(time_data.replace("\t", ",")))
                    for col, value in zip(columns, time_data):
                        data[col].append(value)
            df = pd.DataFrame(data)
            file_save = file[:file.find('.')]+f".{as_type}"
            if save:
                df.to_csv(self.dir_data+f"/{file_save}", index=False)
            else:
                return df

    def extract(self,
                file:str,
                argument: dict):
        """
        :param file:
            :param values: e.g. {"Wind speed at hub": [4,5,6,7,8,9,10]}
        :return:
        """
        id_dot = file.find(".")
        file_name = file[:id_dot]
        file_type = file[id_dot+1:]
        df = pd.read_csv(self.dir_data+f"/{file}", index_col=None)
        column = [*argument][0]
        ids = list()
        for value in argument[column]:
            ids.append(df.loc[df[column]==value].index.tolist()[-1])
        df.iloc[ids, :].to_csv(self.dir_data+f"/{file_name}_extracted_{column}.{file_type}", index=False)