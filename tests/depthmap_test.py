import unittest

import PIL.Image
import numpy as np
from pyproj import Transformer
from scipy.spatial.transform import Rotation


class DepthmapTest(unittest.TestCase):
    def test_uv(self):
        # Real life coordinates
        expected = (-297333.5438620744, 7007741.027668133, 39.84381689661251)

        cu, cv = -1, -1  # Coordinates of the mouse click (UV)
        r, p, y = (-0.253, -40.336, 300.814)
        campos = (-2.66638683195000015, 53.13270710758000348, 241.56999999999999318)
        transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
        campos = transformer.transform(*reversed(campos[:2]), campos[2])
        print(campos)
        # Load the image
        im = PIL.Image.open("../depthmap.tiff")

        # noinspection PyTypeChecker
        im_array = np.array(im)

        size = im.size

        # Convert UV to pixel coordinates
        cpx = int(np.round(((cu + 1) / 2) * size[0]))
        cpy = int(np.round(((cv + 1) / 2) * size[1]))

        # Get the depth from the image
        print(cpx, cpy)
        depth = im_array[cpx, cpy]

        print(depth)
        # Now do our lovely vector maths

        # Generate our vector
        fe = lambda a, x: Rotation.from_euler(a, x, degrees=True).as_matrix()

        rm = fe('z', 360 - y)  # Generate rz - Rotation from yaw
        rm = rm @ fe('y', -r)  # Generate ry - Rotation from roll
        rm = rm @ fe('x', 90 + p)  # Generate rx - Rotation from pitch

        cv = [0, 0, -1]
        cv = rm @ cv

        cv = cv * depth  # Scale the vector
        print(cv)

        final_point = np.add(cv, campos)
        print(final_point)

        # Now apply it so its length is our depth, and its origin is the camera position
        for i, _ in enumerate(expected):
            self.assertEqual(expected[i], final_point[i])


if __name__ == '__main__':
    unittest.main()
