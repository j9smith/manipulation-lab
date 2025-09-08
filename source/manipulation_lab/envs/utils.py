import torch

def reset_env(env, env_ids):
    scene = env.unwrapped.scene
    reset_robot(env, env_ids)

    for object_name in scene.rigid_objects.keys():
        obj = scene.rigid_objects[object_name]
        default_pos = obj.data.default_root_state[env_ids]
        obj.write_root_state_to_sim(default_pos, env_ids=env_ids)

    for object_name in scene.deformable_objects.keys():
        obj = scene.deformable_objects[object_name]
        default_pos = obj.data.default_nodal_state[env_ids]
        obj.write_nodal_state_to_sim(default_pos, env_ids=env_ids)

def reset_robot(env, env_ids):
    scene = env.unwrapped.scene
    robot = scene.articulations["robot"]

    default_root_state = robot.data.default_root_state[env_ids]
    robot.write_root_state_to_sim(default_root_state, env_ids=env_ids)

    default_joint_pos = robot.data.default_joint_pos[env_ids]
    default_joint_vel = robot.data.default_joint_vel[env_ids]

    robot.write_joint_state_to_sim(
        position=default_joint_pos,
        velocity=default_joint_vel,
        env_ids=env_ids
    )

def reset_robot_rand(
        env, env_ids, 
        root_range: tuple[int, int], 
        joint_pos_range: tuple[int, int]
        ):
    
    scene = env.unwrapped.scene
    robot = scene.articulations["robot"]

    root_noise = torch.empty_like(robot.data.default_root_state[env_ids]).uniform_(root_range[0], root_range[1])
    pose_noise = torch.empty_like(robot.data.default_joint_pos[env_ids]).uniform_(joint_pos_range[0], joint_pos_range[1])

    root_state = robot.data.default_root_state[env_ids] + root_noise
    pose_state = robot.data.default_joint_pos[env_ids] + pose_noise

    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        position=pose_state,
        velocity = robot.data.default_joint_vel[env_ids],
        env_ids=env_ids
        )
