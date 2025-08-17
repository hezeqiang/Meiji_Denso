import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf


def draw_arrow(
    robot,                       # meshcat.Visualizer or sub-path
    name: str,                 # subtree name, e.g. "contact_force"
    x_length=1.0, y_length=1.0, z_length=1.0, XY_cross=1.0,  # lengths of the axes
    pose_matrix=np.eye(4),   # pose matrix (world frame)
):
    # Add coordinate frame arrows at the origin using cylinder + cone
    # X-axis arrow (red) - shaft
    robot.viz.viewer["world/"+name+"/x_axis_shaft"].set_object(
        g.Cylinder(height=0.15*x_length, radius=0.005),
        g.MeshLambertMaterial(color=0xff0000)
    )
    robot.viz.viewer["world/"+name+"/x_axis_shaft"].set_transform(
        pose_matrix @  tf.translation_matrix(np.array([0.075, 0.0, 0.0])*x_length) @ tf.rotation_matrix(-np.pi/2, [0, 0, 1])
    )

    # X-axis arrow (red) - head
    robot.viz.viewer["world/"+name+"/x_axis_head"].set_object(
        g.Cylinder(height=0.05*x_length, radiusTop=0, radiusBottom=0.015),
        g.MeshLambertMaterial(color=0xff0000)
    )
    robot.viz.viewer["world/"+name+"/x_axis_head"].set_transform(
        pose_matrix @ tf.translation_matrix(np.array([0.175, 0.0, 0.0])*x_length)  @ tf.rotation_matrix(-np.pi/2, [0, 0, 1])
    )

    # Y-axis arrow (green) - shaft
    robot.viz.viewer["world/"+name+"/y_axis_shaft"].set_object(
        g.Cylinder(height=0.15*y_length, radius=0.005),
        g.MeshLambertMaterial(color=0x00ff00)
    )
    robot.viz.viewer["world/"+name+"/y_axis_shaft"].set_transform(
        pose_matrix @ tf.translation_matrix(np.array([0.0, 0.075, 0.0])*y_length)
    )

    # Y-axis arrow (green) - head
    robot.viz.viewer["world/"+name+"/y_axis_head"].set_object(
        g.Cylinder(height=0.05*y_length, radiusTop=0, radiusBottom=0.015),
        g.MeshLambertMaterial(color=0x00ff00)
    )
    robot.viz.viewer["world/"+name+"/y_axis_head"].set_transform(
        pose_matrix @  tf.translation_matrix(np.array([0.0, 0.175, 0.0])*y_length)
    )

    # Z-axis arrow (blue) - shaft
    robot.viz.viewer["world/"+name+"/z_axis_shaft"].set_object(
        g.Cylinder(height=0.15*z_length, radius=0.005),
        g.MeshLambertMaterial(color=0x0000ff)
    )
    robot.viz.viewer["world/"+name+"/z_axis_shaft"].set_transform(
        pose_matrix @ tf.translation_matrix(np.array([0.0, 0.0, 0.075])*z_length)  @ tf.rotation_matrix(np.pi/2, [1, 0, 0])
    )

    # Z-axis arrow (blue) - head
    robot.viz.viewer["world/"+name+"/z_axis_head"].set_object(
        g.Cylinder(height=0.05*z_length, radiusTop=0, radiusBottom=0.015),
        g.MeshLambertMaterial(color=0x0000ff)
    )
    robot.viz.viewer["world/"+name+"/z_axis_head"].set_transform(
        pose_matrix @ tf.translation_matrix(np.array([0.0, 0.0, 0.175])*z_length) @ tf.rotation_matrix(np.pi/2, [1, 0, 0])
    )
    #     tf.translation_matrix(pose_translation+[0, 0, 0.175])
    # )


    # XY coupling-axis arrow (blue) - shaft
    robot.viz.viewer["world/"+name+"/xy_coupling_axis_shaft"].set_object(
        g.Cylinder(height=0.15*XY_cross, radius=0.005),
        g.MeshLambertMaterial(color=0x0000ff)
    )
    robot.viz.viewer["world/"+name+"/xy_coupling_axis_shaft"].set_transform(
        pose_matrix @ tf.translation_matrix(np.array([0.053, 0.053, 0])*XY_cross) @ tf.rotation_matrix(-np.pi/4, [0, 0, 1])
    )


    # XY coupling-axis arrow (blue) - head
    robot.viz.viewer["world/"+name+"/xy_coupling_axis_head"].set_object(
        g.Cylinder(height=0.05*XY_cross, radiusTop=0, radiusBottom=0.015),
        g.MeshLambertMaterial(color=0x0000ff)
    )
    robot.viz.viewer["world/"+name+"/xy_coupling_axis_head"].set_transform(
        pose_matrix @ tf.translation_matrix(np.array([0.1237, 0.1237, 0])*XY_cross) @ tf.rotation_matrix(-np.pi/4, [0, 0, 1])
    )
    #     tf.translation_matrix(pose_translation+[0, 0, 0.175])
    # )


