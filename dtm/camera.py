import numpy as np
import trimesh
from trimesh.creation import camera_marker
from trimesh.scene import Camera

from dtm.image import Image


class DTCamera(Camera):
    image: Image
    coords: (int, int)
    z_offset: int = 0

    def __init__(self, image: Image,
                 coords: (int, int) = None,
                 focal=None,
                 z_offset=0,
                 z_near=0.01,
                 z_far=10000.0
                 ):
        super(DTCamera, self).__init__(f"Camera: {image.uri}", image.resolution, focal, [image.fov, image.fov],
                                       z_near, z_far)

        self.image = image
        self.coords = coords

    @property
    def cam_pt(self):
        return *self.coords, self.image.campos[2] + self.z_offset

    @property
    def marker(self) -> [trimesh.Trimesh]:
        ma = np.zeros((4, 4))
        ma[:3, :3] = self.image.rs_matrix()
        ma[3, 3] = 1

        meshes: [trimesh.Trimesh] = camera_marker(self, marker_height=100)

        for m in meshes:
            m.apply_transform(-ma)
            m.apply_translation(self.cam_pt)

        return meshes
