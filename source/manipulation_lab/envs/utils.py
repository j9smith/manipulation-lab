def reset_env(env, env_ids):
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

    for object_name in scene.rigid_objects.keys():
        obj = scene.rigid_objects[object_name]
        default_pos = obj.data.default_root_state[env_ids]
        obj.write_root_state_to_sim(default_pos, env_ids=env_ids)
