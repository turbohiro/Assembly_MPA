import pybullet as p
import os

p.connect(p.DIRECT)
name_in ="franka/urdf/meshes/visual/finger.obj"
name_out = "franka/urdf/meshes/visual/finger_vhacd.obj"
name_log = "log.txt"
p.vhacd(name_in, name_out, name_log, alpha=0.0001,resolution=1000000)
#test
#dir2 = 'mug/collision/'
#for home, dirs, files in os.walk('mug/visual/'):
#    print('hhhh', files)
#    for filename in files:
 #       name_in = os.path.join(home, filename)
#        name_out = os.path.join(dir2, filename.split('.')[0] + '_vhacd.obj')
#        name_log = "log.txt"
#        p.vhacd(name_in, name_out, name_log, alpha=0.04, resolution=100000)
