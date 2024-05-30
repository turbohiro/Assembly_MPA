import trimesh

for i in range(32):
    mesh = trimesh.load('peg_circle36_three_collision_'+str(i)+'.obj')
    mesh.export('peg_circle36_three_collision_'+str(i)+'.stl', file_type="stl")

#how to make collision model for arbitary stl/CAD model

#1.python stl2obj.py
#2.obj2mjcf --obj-dir . --obj-filter cup --save-mjcf --compile-model --vhacd-args.enable
#3.  ./simulate /data/robosuite_manipulation/robosuite/me/obj2mjcf/tests/cup/cup.xml