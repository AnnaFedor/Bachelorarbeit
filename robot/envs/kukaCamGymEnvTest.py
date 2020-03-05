# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from kukaCamGymEnv import KukaCamGymEnv
import time


def main():
    environment = KukaCamGymEnv(renders=True, isDiscrete=True)

    motorsIds = []
    # motorsIds.append(environment._p.addUserDebugParameter("posX",0.4,0.75,0.537))
    # motorsIds.append(environment._p.addUserDebugParameter("posY",-.22,.3,0.0))
    # motorsIds.append(environment._p.addUserDebugParameter("posZ",0.1,1,0.2))
    # motorsIds.append(environment._p.addUserDebugParameter("yaw",-3.14,3.14,0))
    # motorsIds.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,.3))

    dv = 1
    motorsIds.append(environment._p.addUserDebugParameter("posX", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("posY", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("posZ", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("yaw", -dv, dv, 0))
    motorsIds.append(environment._p.addUserDebugParameter("fingerAngle", 0, 0.3, .3))

    done = False
    # environment.changeXYCoordinates(0.64, 0.2)
    print()
    # print("cartesian coordinates changed")
    print()
    time.sleep(4)
    """
    while not done:
        action = []
        for motorId in motorsIds:
            action.append(environment._p.readUserDebugParameter(motorId))
        # state, reward, done, info = environment.goDown()
        done = True
        # obs = environment.getExtendedObservation()
    done = False
    """
    # state, reward, done, info = environment.changeXYCoordinates(0.6, 0.3)
    # state, reward, done, info = environment.goDown(0)
    # print("state: ", state)
    state, reward, done, info = environment.step((2, 4))
    print("action type: ", environment.get_action_type())
    print("After the grasp attempt: ")
    print("pos: ", environment.getEndEffectorPosition())
    print("actualPos: ", environment.getActualEndEffectorPosition())
    print("reward: ", reward)
    print("done: ", done)
    print("info: ", info)
    time.sleep(2)


if __name__ == "__main__":
    main()
