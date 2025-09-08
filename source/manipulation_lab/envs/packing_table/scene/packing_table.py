from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

class PackingTableSceneCfg:
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/PackingTable",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg()
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.7071, 0.0, 0.0, -0.7071)
        )
    )
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.5, 0.5, 0.5)),
    )