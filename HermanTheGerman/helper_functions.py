import pathlib
import os
import shutil
import matplotlib

class Helper():
    def __init__(self):
        pass

    def create_dir(self,
                   path_dir: str,
                   overwrite: bool = False,
                   add_missing_parent_dirs: bool = True,
                   raise_exception: bool = False,
                   print_message: bool = False) \
            -> tuple[str, bool]:
        return self._create_dir(path_dir, overwrite, add_missing_parent_dirs, raise_exception, print_message)

    @staticmethod
    def _create_dir(target: str,
                     overwrite: bool,
                     add_missing_parent_dirs: bool,
                     raise_exception: bool,
                     print_message: bool) \
            -> tuple[str, bool]:
        msg, keep_going = str(), bool()
        try:
            if overwrite:
                if os.path.isdir(target):
                    shutil.rmtree(target)
                    msg = f"Existing directory {target} was overwritten."
                else:
                    msg = f"Could not overwrite {target} as it did not exist. Created it instead."
                keep_going = True
            else:
                msg, keep_going = f"Directory {target} created successfully.", True
            pathlib.Path(target).mkdir(parents=add_missing_parent_dirs, exist_ok=False)
        except Exception as exc:
            if exc.args[0] == 2:  # FileNotFoundError
                if raise_exception:
                    raise FileNotFoundError(f"Not all parent directories exist for directory {target}.")
                else:
                    msg, keep_going = f"Not all parent directories exist for directory {target}.", False
            elif exc.args[0] == 17:  # FileExistsError
                if raise_exception:
                    raise FileExistsError(f"Directory {target} already exists and was not changed.")
                else:
                    msg, keep_going = f"Directory {target} already exists and was not changed.", False
        if print_message:
            print(msg)
        return msg, keep_going

    @staticmethod
    def handle_figure(figure: matplotlib.figure,
                      file_figure: str=False,
                      show: bool=False,
                      size: tuple=(18.5, 10),
                      inches: int=100,
                      tight_layout: bool=True) -> matplotlib.figure:
        figure.set_size_inches(size)
        figure.set_dpi(inches)
        figure.set_tight_layout(tight_layout)
        if file_figure:
            figure.savefig(file_figure)
        if show:
            figure.show()
        return figure

    @staticmethod
    def handle_axis(axis: matplotlib.pyplot.axis,
                    title: str=None,
                    grid: bool=False,
                    legend: bool=False,
                    legend_columns: int=1,
                    x_label: str=False,
                    y_label: str=False,
                    z_label: str=False,
                    x_scale: str="linear",
                    y_scale: str="linear",
                    font_size: int=False) -> matplotlib.pyplot.axis:
        axis.set_title(title)
        if font_size:
            axis.title.set_fontsize(font_size)
            axis.xaxis.label.set_fontsize(font_size)
            axis.yaxis.label.set_fontsize(font_size)
            if axis.name == "3d":
                axis.zaxis.label.set_fontsize(font_size)
            axis.tick_params(axis='both', labelsize=font_size)
        axis.grid(grid)
        if legend:
            axis.legend(ncol=legend_columns, prop={"size": font_size}) if font_size else axis.legend(ncol=legend_columns)
        if x_label:
            axis.set_xlabel(x_label, labelpad=20)
        if y_label:
            axis.set_ylabel(y_label, labelpad=20)
        if z_label:
            axis.set_zlabel(z_label, labelpad=40)
        axis.set_xscale(x_scale)
        axis.set_yscale(y_scale)
        return axis

