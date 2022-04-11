import numpy as np
from scipy.spatial.transform.rotation import Rotation


class Image:
    uri: str
    campos: (float, float, float)  # [x,y,z]
    camrpy: (float, float, float)  # [r,p,y]
    fov: float
    # distortion: float  # This comes later

    height: int
    width: int

    def __init__(self, config):
        def clean_geom(string: str):
            string = string.replace("[", '').replace("]", "")
            return [float(v) for v in string.split(" ")]

        self.uri = config.get("file_name")
        self.campos = clean_geom(config.get("wkt_geom"))
        self.vppos = clean_geom(config.get("vp_geom"))
        self.camrpy = [float(config.get(v)) for v in ["roll", "pitch", "yaw"]]
        self.width = int(config.get("x_pixels"))
        self.height = int(config.get("y_pixels"))
        self.fov = float(config.get("fov"))

    def rs_matrix(self):
        roll, pitch, yaw = self.camrpy
        fe = lambda a, x: Rotation.from_euler(a, x, degrees=True).as_matrix()

        # print(pitch, roll, yaw)
        rm = fe('z', 360 - yaw)  # Generate rz - Rotation from yaw
        rm = rm @ fe('y', -roll)  # Generate ry - Rotation from roll
        rm = rm @ fe('x', 90 + pitch)  # Generate rx - Rotation from pitch

        # rm = fe('x', np.sqrt(2) * pitch)  # Generate rx - Rotation from pitch
        # rm = fe('x', np.sqrt(2) * pitch)  # Generate rx - Rotation from pitch
        # rm = np.matmul(fe('y', -roll), rm)  # Generate ry - Rotation from roll
        # rm = np.matmul(fe('z', yaw-45), rm)  # Generate rz - Rotation from yaw

        # rm = np.matmul(rm, self.campos)

        return rm

    @property
    def aspect(self):
        return self.width / self.height

    @property
    def resolution(self):
        return self.width, self.height

    @property
    def focal(self):
        return np.arctan(self.fov / 2)
