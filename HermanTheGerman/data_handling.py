import pandas as pd
import scipy.interpolate as interpolate


class AirfoilInterpolator:
    def __init__(self,
                 dir_data: str,
                 data_profiles: dict,
                 **pandas_read_csv):
        self.dir_data = dir_data
        self.df_profiles = dict()
        for specific, file in data_profiles.items():
            self.df_profiles[specific] = pd.read_csv(dir_data+"/"+file, **pandas_read_csv)

    def interpolate(self,
                    to_interpolate: list or str,
                    second_argument: str) -> dict:
        if type(to_interpolate) == str:
            to_interpolate = [to_interpolate]

        points = {var: list() for var in to_interpolate}
        values = {var: list() for var in to_interpolate}
        for base_arg in self.df_profiles.keys():
            for _, row in self.df_profiles[base_arg].iterrows():
                for variable in to_interpolate:
                    points[variable].append((base_arg, row[second_argument]))
                    values[variable].append(row[variable])

        interpolated = {var: None for var in to_interpolate}
        for variable in to_interpolate:
            interpolated[variable] = interpolate.LinearNDInterpolator(points[variable], values[variable])
        return interpolated




