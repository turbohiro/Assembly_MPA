import trimesh

# for i in range(31):
#     mesh = trimesh.load('peg_circle_'+str(i)+'.obj')
#     mesh.export('peg_circle_'+str(i)+'.stl', file_type="stl")
mesh = trimesh.load('circle_collision_31.obj')
mesh.export('circle_collision_31.stl', file_type="stl")
#how to make collision model for arbitary stl/CAD model

#1.python  python stl2obj.py
#2.obj2mjcf --obj-dir . --obj-filter cup --save-mjcf --compile-model --vhacd-args.enable
#3.  ./simulate /data/robosuite_manipulation/robosuite/me/obj2mjcf/tests/cup/cup.xml