import trimesh

mesh = trimesh.load("peg_circle.stl")
mesh.export("peg_circle.obj", file_type="obj")

#how to make collision model for arbitary stl/CAD model

#1.python  python stl2obj.py
#2.obj2mjcf --obj-dir . --obj-filter cup --save-mjcf --compile-model --vhacd-args.enable
#3.  ./simulate /data/robosuite_manipulation/robosuite/me/obj2mjcf/tests/cup/cup.xml