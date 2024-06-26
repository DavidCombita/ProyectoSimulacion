from utils.multi_robot_plot import pintarRobotYObstaculos
from utils.create_obstacles import CrearObstaculos
from utils.control import CalcularVelocidadDeseada
import numpy as np

SIM_TIME = 5.
TIMESTEP = 0.1
NUMBER_OF_TIMESTEPS = int(SIM_TIME/TIMESTEP)
ROBOT_TAMANIO = 0.5
VMAX = 2
VMIN = 0.2


def simulate():
    obstacles = CrearObstaculos(SIM_TIME, NUMBER_OF_TIMESTEPS)

    inicio = np.array([5, 0, 0, 0])
    final = np.array([5, 5, 0, 0])

    estadoRobot = inicio
    estadoRobotHistorial = np.empty((4, NUMBER_OF_TIMESTEPS))
    for i in range(NUMBER_OF_TIMESTEPS):
        v_desired = CalcularVelocidadDeseada(estadoRobot, final, ROBOT_TAMANIO, VMAX)
        control_vel = CalcularVelocidad(
            estadoRobot, obstacles[:, i, :], v_desired)
        estadoRobot = ActualizarEstado(estadoRobot, control_vel)
        estadoRobotHistorial[:4, i] = estadoRobot

    pintarRobotYObstaculos(
        estadoRobotHistorial, obstacles, ROBOT_TAMANIO, NUMBER_OF_TIMESTEPS, SIM_TIME)


def CalcularVelocidad(robot, obstacles, v_desired):
    pA = robot[:2]
    vA = robot[-2:]
    # Compute the constraints
    # for each velocity obstacles
    number_of_obstacles = np.shape(obstacles)[1]
    Amat = np.empty((number_of_obstacles * 2, 2))
    bvec = np.empty((number_of_obstacles * 2))
    for i in range(number_of_obstacles):
        obstacle = obstacles[:, i]
        pB = obstacle[:2]
        vB = obstacle[2:]
        dispBA = pA - pB
        distBA = np.linalg.norm(dispBA)
        thetaBA = np.arctan2(dispBA[1], dispBA[0])
        if 2.2 * ROBOT_TAMANIO > distBA:
            distBA = 2.2*ROBOT_TAMANIO
        phi_obst = np.arcsin(2.2*ROBOT_TAMANIO/distBA)
        phi_left = thetaBA + phi_obst
        phi_right = thetaBA - phi_obst

        # VO
        translation = vB
        Atemp, btemp = CrearBarreras(translation, phi_left, "left")
        Amat[i*2, :] = Atemp
        bvec[i*2] = btemp
        Atemp, btemp = CrearBarreras(translation, phi_right, "right")
        Amat[i*2 + 1, :] = Atemp
        bvec[i*2 + 1] = btemp

    # Create search-space
    th = np.linspace(0, 2*np.pi, 20)
    vel = np.linspace(0, VMAX, 5)

    vv, thth = np.meshgrid(vel, th)

    vx_sample = (vv * np.cos(thth)).flatten()
    vy_sample = (vv * np.sin(thth)).flatten()

    v_sample = np.stack((vx_sample, vy_sample))

    v_satisfying_constraints = VerificarMuros(v_sample, Amat, bvec)

    # Objective function
    size = np.shape(v_satisfying_constraints)[1]
    diffs = v_satisfying_constraints - \
        ((v_desired).reshape(2, 1) @ np.ones(size).reshape(1, size))
    norm = np.linalg.norm(diffs, axis=0)
    min_index = np.where(norm == np.amin(norm))[0][0]
    cmd_vel = (v_satisfying_constraints[:, min_index])

    return cmd_vel


def VerificarMuros(v_sample, Amat, bvec):
    length = np.shape(bvec)[0]

    for i in range(int(length/2)):
        v_sample = VerificarInteriror(v_sample, Amat[2*i:2*i+2, :], bvec[2*i:2*i+2])

    return v_sample


def VerificarInteriror(v, Amat, bvec):
    v_out = []
    for i in range(np.shape(v)[1]):
        if not ((Amat @ v[:, i] < bvec).all()):
            v_out.append(v[:, i])
    return np.array(v_out).T


def CrearBarreras(translation, angle, side):
    # create line
    origin = np.array([0, 0, 1])
    point = np.array([np.cos(angle), np.sin(angle)])
    line = np.cross(origin, point)
    line = MoverLinea(line, translation)

    if side == "left":
        line *= -1

    A = line[:2]
    b = -line[2]

    return A, b


def MoverLinea(line, translation):
    matrix = np.eye(3)
    matrix[2, :2] = -translation[:2]
    return matrix @ line


def ActualizarEstado(x, v):
    new_state = np.empty((4))
    new_state[:2] = x[:2] + v * TIMESTEP
    new_state[-2:] = v
    return new_state
