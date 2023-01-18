import pybullet as p
import os

p.connect(p.DIRECT)
name_in ="circle.obj"
name_out = "circle_vhacd.obj"
name_log = "log.txt"
p.vhacd(name_in, name_out, name_log, alpha=0.001,resolution=500000 )