from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

@configclass
class WristCameraCfg(CameraCfg): 
    prim_path: str
    offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(pos=(0.1, 0.0, -0.06), #x = vertical, y = horizontal, z = forwards(+)/backwards(-)
                                   rot=(0.0, 0.60, 0.0, 0.80),
                                   convention="world")
    data_types: str = ["rgb"]
    spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
            focal_length=10, 
            focus_distance=400.0, 
            horizontal_aperture=10,
            clipping_range=(0.1, 1.0e5))
    width: int = 256
    height: int = 256
