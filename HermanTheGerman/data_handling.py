import numpy as np
import pandas as pd
import scipy.interpolate as interpolate


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
        Interpolation
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




