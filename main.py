import mujoco as mj
import mujoco_viewer
from mujoco.glfw import glfw
import numpy as np
import os

xml_path = 'models/car_wall.xml'


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    abspath = os.path.join(dirname + "/" + xml_path)
    xml_path = abspath

    # MuJoCo data structures
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    options = mj.MjvOption()
    mj.mjv_defaultOption(options)
    options.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    options.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    options.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # create the viewer object
    viewer = mujoco_viewer.MujocoViewer(model, data)
    data.ctrl = [0.5]
    # simulate and render
    for _ in range(10000):
        if viewer.is_alive:
            # if _ > 20:
            #     data.qvel[0] = 1
            mj.mj_step(model, data)
            viewer.render()
        else:
            break

    # close
    viewer.close()
