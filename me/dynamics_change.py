import robosuite as suite
from robosuite.utils.mjmod import DynamicsModder
import numpy as np

# Create environment and modder
env = suite.make("Lift", robots="Panda")
modder = DynamicsModder(sim=env.sim, random_state=np.random.RandomState(5))

# Define function for easy printing
cube_body_id = env.sim.model.body_name2id(env.cube.root_body)
cube_geom_ids = [env.sim.model.geom_name2id(geom) for geom in env.cube.contact_geoms]

def print_params():
    print(f"cube mass: {env.sim.model.body_mass[cube_body_id]}")
    print(f"cube frictions: {env.sim.model.geom_friction[cube_geom_ids]}")
    print()

# Print out initial parameter values
print("INITIAL VALUES")
print_params()

# Modify the cube's properties
modder.mod(env.cube.root_body, "mass", 5.0)                                # make the cube really heavy
for geom_name in env.cube.contact_geoms:
    modder.mod(geom_name, "friction", [2.0, 0.2, 0.04])           # greatly increase the friction
modder.update()                                                   # make sure the changes propagate in sim

# Print out modified parameter values
print("MODIFIED VALUES")
print_params()

# We can also restore defaults (original values) at any time
modder.restore_defaults()

# Print out restored initial parameter values
print("RESTORED VALUES")
print_params()