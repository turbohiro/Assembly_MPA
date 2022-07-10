import os
import open3d as o3d
import pdb
def get_urdf_str(mesh_dir1, mesh_dir2, scale_values):
    urdf_str = ("""
<robot name="model.urdf">
  <link name="base_link">
    <contact>
      <friction_anchor/>
      <lateral_friction value="0.8"/>
      <spinning_friction value="0.001"/>
      <rolling_friction value="0.001"/>
      <contact_cfm value="0.1"/>
      <contact_erp value="1.0"/>
    </contact>
    <inertial>
      <origin xyz="0 0 0" />
      <mass value="0.1" />
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <mesh filename="%s" scale="%s %s %s"/>
      </geometry>
      <material name="white">
        <color rgba="1. 1. 1. 1."/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <mesh filename="%s" scale="%s %s %s"/>
      </geometry>
    </collision>
  </link>
</robot>
""") % (mesh_dir1, scale_values[0], scale_values[1], scale_values[2], mesh_dir2, scale_values[0], scale_values[1],
        scale_values[2])
    return urdf_str


def main():
    scale_values = [0.1, 0.1, 0.1]

    dir2 = 'mug/collision/'
    for home, dirs, files in os.walk('mug/visual/'):
        print('hhhh', files)

        for filename in files:
            fullname = os.path.join(home, filename)
            fullname2 = os.path.join(dir2, filename.split('.')[0] + '_vhacd.obj')

            urdf_str = get_urdf_str(fullname, fullname2, scale_values)

            obj_dir = 'urdf/'

            os.makedirs(obj_dir, exist_ok=True)
            f = open(os.path.join(obj_dir, filename.split('.')[0] + '.urdf'), 'w')

            f.write(urdf_str)
            # print("urdf model saved in {}".format(os.path.join(obj_dir,"model.urdf")))

def generate_pcd():
  
  for home, dirs, files in os.walk('mug/visual/'):
      print('hhhh', files)

      for filename in files:
          pdb.set_trace()
          fullname = os.path.join(home, filename)
          mesh = o3d.io.read_triangle_mesh(filename)
          pointcloud = mesh.sample_points_poisson_disk(2048)
          target_dir = ('mug/grasp/%s.pcd')%(filename)
          o3d.io.write_point_cloud(target_dir, pointcloud)

if __name__ == "__main__":
    #main()
    generate_pcd()
