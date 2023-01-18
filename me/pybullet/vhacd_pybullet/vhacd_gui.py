import pybullet as p
import pybullet_data as pd
import os
import time
import json
import jsbeautifier
import numpy as np
import math
import geom_utils
import argparse
dir=""
timestr = time.strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--urdf_file', help='Video file', type=str, default=dir+"circle.obj")


args = parser.parse_args()

opts = jsbeautifier.default_options()
opts.indent_size = 2

p.connect(p.GUI)#DIRECT)

name_in = args.urdf_file
name_out = args.urdf_file+"_vhacd.obj"
name_log = dir+timestr+name_in+"log.txt"

#for more parameters, see http://pybullet.org/guide
  

for depth in [4]:
  p.vhacd(name_in, name_out, name_log, resolution=500000, depth=depth)

  meshScale = [1, 1, 1]
  collisionShapeId_VHACD = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                          fileName=name_out,
                                        flags=p.URDF_INITIALIZE_SAT_FEATURES,
                                          meshScale=meshScale)

  
  body_uid = p.createMultiBody(baseMass=0,
                  baseCollisionShapeIndex=collisionShapeId_VHACD,
                  basePosition=[0,0,0],#2, depth, 1],
                  flags=p.URDF_INITIALIZE_SAT_FEATURES,
                  useMaximalCoordinates=False)
  collision_shapes = p.getCollisionShapeData(body_uid,-1)
  data = {}
  data['target']=name_in
  prims=[]
  for shape_index in range (len(collision_shapes)):
    shape = collision_shapes[shape_index]
    prim={}
    mesh = p.getMeshData(body_uid,-1,shape_index)
    num_verts = mesh[0]
    vertices=mesh[1]
          
    minX = 1e30
    minY = 1e30
    minZ = 1e30
    maxX =-1e30
    maxY =-1e30
    maxZ =-1e30
    
    for v in vertices:
      if v[0]<minX:
        minX=v[0]
      if v[1]<minY:
        minY=v[1]
      if v[2]<minZ:
        minZ=v[2]
      if v[0]>maxX:
        maxX=v[0]
      if v[1]>maxY:
        maxY=v[1]
      if v[2]>maxZ:
        maxZ=v[2]
          
      pos = [(maxX+minX)*0.5,(maxY+minY)*0.5,(maxZ+minZ)*0.5]
      rpy=[0,0,0]
      dim= [(maxX-minX)*0.5,(maxY-minY)*0.5, (maxZ-minZ)*0.5]

    aabb=[]
    aabb.append([pos[0]-dim[0],pos[1]-dim[1],pos[2]-dim[2]])
    aabb.append([pos[0]+dim[0],pos[1]+dim[1],pos[2]+dim[2]])
      
    geom_utils.drawAABB(p,aabb)
    prim["type"] = "box"
    prim["position"] = pos
    prim["orientation"] = rpy
    prim["dimensions"]=dim
    prims.append(prim)   
  data['primitives']=prims
  name = name_out+".json"
  with open(name, 'w') as json_file:
    json_file.write(jsbeautifier.beautify(json.dumps(data), opts))
  
  
  #write a json file
  


#reference concave body
if 0:
  collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                            fileName=name_in,
            flags=p.URDF_INITIALIZE_SAT_FEATURES +p.GEOM_FORCE_CONCAVE_TRIMESH,
                                            meshScale=meshScale)

  collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                            fileName=name_in,
                                          flags=p.URDF_INITIALIZE_SAT_FEATURES +p.GEOM_FORCE_CONCAVE_TRIMESH,
                                            meshScale=meshScale)
  p.createMultiBody(baseMass=0,
                    baseCollisionShapeIndex=collisionShapeId,
                    basePosition=[0, 0, 1],
                    flags=p.URDF_INITIALIZE_SAT_FEATURES,
                    useMaximalCoordinates=False)


#p.loadURDF(pd.getDataPath()+"/plane.urdf")

dt = 1./240.
p.setGravity(0,0,-9.81)

while (1):
  p.stepSimulation()
  time.sleep(dt)

