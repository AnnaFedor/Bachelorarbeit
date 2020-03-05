import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import kuka
# import kuka
import random
import robot.data as data
from pkg_resources import parse_version

maxSteps = 1000

RENDER_HEIGHT = 720
RENDER_WIDTH = 960



class KukaCamGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 urdfRoot=data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False):
        self._timeStep = 1. / 120.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._width = 341
        self._height = 256
        self._isDiscrete = isDiscrete
        self.terminated = 0
        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
        # timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
        self.seed()
        self.reset()
        observationDim = len(self.getExtendedObservation())
        # print("observationDim")
        # print(observationDim)

        observation_high = np.array([np.finfo(np.float32).max] * observationDim)
        if (self._isDiscrete):
            self.action_space = spaces.Tuple((
        spaces.Discrete(20),
        spaces.Discrete(20)))
        else:
            action_dim = 3
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self._height, self._width, 4),
                                            dtype=np.uint8)
        self.viewer = None

    def get_action_type(self):
        action_space = self.action_space
        '''Method to get the action type to choose prob. dist. to sample actions from NN logits output'''
        if isinstance(action_space, spaces.Box):
            shape = action_space.shape
            assert len(shape) == 1
            if shape[0] == 1:
                return 'continuous'
            else:
                return 'multi_continuous'
        elif isinstance(action_space, spaces.Discrete):
            return 'discrete'
        elif isinstance(action_space, spaces.MultiDiscrete):
            return 'multi_discrete'
        elif isinstance(action_space, spaces.MultiBinary):
            return 'multi_binary'
        else:
            raise NotImplementedError
        
    def reset(self):
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        self.tableid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                                  0.000000, 0.000000, 0.0, 1.0)

        xpos = 0.5 + 0.2 * random.random()
        ypos = 0 + 0.25 * random.random()
        ang = 3.1415925438 * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])
        self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), basePosition=(xpos, ypos, -0.1),
                                   baseOrientation=(orn[0], orn[1], orn[2], orn[3]), globalScaling=3)
        print("in the process of resetting")

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):

        """
        # camEyePos = [0.03,0.236,0.54]
        distance = 0.06
        pitch=-56
        yaw = 258
        roll=0
        upAxisIndex = 2"""
        camEyePos = [0.55, 0, 0.1]
        distance = 0.06
        pitch = -90
        yaw = 180
        roll = 0
        upAxisIndex = 2
        # camInfo = p.getDebugVisualizerCamera()
        # print(camInfo)
        # print("here")
        # time.sleep(4)
        # print("width,height")
        # print(camInfo[0])
        # print(camInfo[1])
        # print("viewMatrix")
        # print(camInfo[2])
        # print("projectionMatrix")
        # print(camInfo[3])
        # viewMat = camInfo[2]
        viewMat = p.computeViewMatrixFromYawPitchRoll(camEyePos, distance, yaw, pitch, roll, upAxisIndex)
        """viewMat = [
            -0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722,
            -0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843,
            0.8348482847213745, 0.0, 0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0
        ]"""
        projMatrix = [
            0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
            -0.02000020071864128, 0.0
        ]
        """
        projMatrix = p.computeProjectionMatrix(-1, 1, -0.5, 0.5, 0.7, 0, 0)"""

        img_arr = p.getCameraImage(width=self._width,
                                   height=self._height,
                                   viewMatrix=viewMat,
                                   projectionMatrix=projMatrix)
        rgb = img_arr[2]
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        self._observation = np_img_arr
        return self._observation

    def getEndEffectorPosition(self):
        return self._kuka.endEffectorPos

    def getActualEndEffectorPosition(self):
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]
        return actualEndEffectorPos

    def getPositionalDifference(self):
        endEffectorPos = self.getEndEffectorPosition()
        actualEndEffectorPos = self.getActualEndEffectorPosition()
        difference = [endEffectorPos[0] - actualEndEffectorPos[0], endEffectorPos[1] - actualEndEffectorPos[1],
                      endEffectorPos[2] - actualEndEffectorPos[2]]
        return difference

    def changeXYCoordinates(self, x, y):
        endEffectorPos = self.getEndEffectorPosition()
        actualEndEffectorPos = self.getActualEndEffectorPosition()
        targetPosition = [x, y, endEffectorPos[2]]
        positionalDifference = self.getPositionalDifference()
        neverDone = True
        while neverDone or abs(positionalDifference[0]) > 0.1 or abs(positionalDifference[1]) > 0.1:
            self._kuka.applyAction1(targetPosition)
            p.stepSimulation()
            # self._observation = self.getExtendedObservation()
            self._envStepCounter += 1
            positionalDifference = self.getPositionalDifference()
            # print("difference: ", positionalDifference)
            neverDone = False

        self._observation = self.getExtendedObservation()
        if self._renders:
            time.sleep(self._timeStep)


    def goDown(self, value):
        # for i in range(self._actionRepeat):
        print("going down")
        endEffectorPos = self.getEndEffectorPosition()
        targetPosition = [endEffectorPos[0], endEffectorPos[1], value]
        positionalDifference = self.getPositionalDifference()
        neverDone = True
        while neverDone or abs(positionalDifference[2]) > 0.05:
            self._kuka.applyAction1(targetPosition)
            p.stepSimulation()
            if self._termination():
                break
            # self._observation = self.getExtendedObservation()
            self._envStepCounter += 1
            positionalDifference = self.getPositionalDifference()
            # print("difference: ", positionalDifference)
            neverDone = False

        self._observation = self.getExtendedObservation()
        if self._renders:
            time.sleep(self._timeStep)

        # print("self._envStepCounter")
        # print(self._envStepCounter)

        done = self._termination()
        reward = self._reward()
        # print("len=%r" % len(self._observation))
        return np.array(self._observation), reward, done, {}
    # Step function for continiuos action space (kind of, don’t think it works right xD)
    """
    def step(self, action):
        xTarget = action[0]
        yTarget = action[1]
        self.changeXYCoordinates(xTarget, yTarget)
        state, reward, done, info = self.goDown(0)
        return state, reward, done, info
    """

    def step(self, action):
        print("action: ", action)
        if (self._isDiscrete):
            # dx, dy - Step for x, y Coordinates
            dx = 0.015
            dy = 0.02
            xTarget = 0.4 + action[0]*dx
            yTarget = -0.2 + action[1]*dy
            # da = [0, 0, 0, 0, 0, -0.1, 0.1][action]
            # f = 0.3
            realAction = [xTarget, yTarget]
        else:
            dv = 0.01
            dx = action[0] * dv
            dy = action[1] * dv
            da = action[2] * 0.1
            f = 0.3
            realAction = [dx, dy, -0.002, da, f]

        return self.step2(realAction)

    def step2(self, action):
        xTarget = action[0]
        yTarget = action[1]
        print("Target position: ", xTarget, " ", yTarget)
        self.changeXYCoordinates(xTarget, yTarget)
        state, reward, done, info = self.goDown(0)
        return state, reward, done, info

    def step5(self, action):
        if (self._isDiscrete):
            dv = 0.01
            dx = [0, -dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0][action]
            da = [0, 0, 0, 0, 0, -0.1, 0.1][action]
            f = 0.3
            realAction = [dx, dy, -0.002, da, f]
        else:
            dv = 0.01
            dx = action[0] * dv
            dy = action[1] * dv
            da = action[2] * 0.1
            f = 0.3
            realAction = [dx, dy, -0.002, da, f]
        # print("111111111111")

        return self.step2(realAction)
    
    """
    def step2(self, action):
        # for i in range(self._actionRepeat):
        positionalDifference = self.getPositionalDifference()
        neverDone = True
        while neverDone or abs(positionalDifference[0]) > 0.1 or abs(positionalDifference[1]) > 0.1:
            self._kuka.applyAction1(action)
            p.stepSimulation()
            if self._termination():
                break
            # self._observation = self.getExtendedObservation()
            self._envStepCounter += 1
            positionalDifference = self.getPositionalDifference()
            # print("difference: ", positionalDifference)
            neverDone = False

        self._observation = self.getExtendedObservation()
        if self._renders:
            time.sleep(self._timeStep)

        # print("self._envStepCounter")
        # print(self._envStepCounter)

        done = self._termination()
        reward = self._reward()
        # print("len=%r" % len(self._observation))
        return np.array(self._observation), reward, done, {}
    """
    
    def step3(self, action):
        for i in range(self._actionRepeat):
            self._kuka.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            # self._observation = self.getExtendedObservation()
            self._envStepCounter += 1

        self._observation = self.getExtendedObservation()
        if self._renders:
            time.sleep(self._timeStep)

        # print("self._envStepCounter")
        # print(self._envStepCounter)

        done = self._termination()
        reward = self._reward()
        # print("len=%r" % len(self._observation))

        return np.array(self._observation), reward, done, {}

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                  height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        # print (self._kuka.endEffectorPos[2])
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]

        # print("self._envStepCounter")
        # print(self._envStepCounter)
        if self.terminated or self._envStepCounter > maxSteps:
            self._observation = self.getExtendedObservation()
            return True
        maxDist = 0.005
        closestPoints = p.getClosestPoints(self.tableid, self._kuka.kukaUid, maxDist)

        if len(closestPoints):  # (actualEndEffectorPos[2] <= -0.43):
            self.terminated = 1


            #print()
            #print()
            #print("closing gripper, attempting grasp")
            # start grasp and terminate
            fingerAngle = 0.3
            for i in range(100):
                graspAction = [0, 0, 0.0001, 0, fingerAngle]
                self._kuka.applyAction(graspAction)
                p.stepSimulation()
                fingerAngle = fingerAngle - (0.3 / 100.)
                if (fingerAngle < 0):
                    fingerAngle = 0

            for i in range(1000):
                graspAction = [0, 0, 0.001, 0, fingerAngle]
                self._kuka.applyAction(graspAction)
                p.stepSimulation()
                blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
                if (blockPos[2] > 0.23):
                    # print("BLOCKPOS!")
                    # print(blockPos[2])
                    break
                state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
                actualEndEffectorPos = state[0]
                if (actualEndEffectorPos[2] > 0.5):
                    break

            self._observation = self.getExtendedObservation()
            return True
        return False

    def _reward(self):

        # rewards is height of target object
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        closestPoints = p.getClosestPoints(self.blockUid, self._kuka.kukaUid, 1000, -1,
                                           self._kuka.kukaEndEffectorIndex)

        reward = -1000
        numPt = len(closestPoints)
        # print(numPt)
        if (numPt > 0):
            # print("reward:")
            reward = -closestPoints[0][8] * 10
        if (blockPos[2] > 0.2):
            # print("grasped a block!!!")
            # print("self._envStepCounter")
            # print(self._envStepCounter)
            reward = reward + 1000

        # print("reward")
        # print(reward)
        return reward

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step
