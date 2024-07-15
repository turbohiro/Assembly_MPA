from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion

class AssemblyArena(Arena):

    def __init__(self,seed_object):
        arena_list =["arenas/circle_arena.xml","arenas/ellipse_arena.xml","arenas/triangle_arena.xml","arenas/square_arena.xml"]
        #import random
        arena_path = arena_list[seed_object]
        super().__init__(xml_path_completion(arena_path))
