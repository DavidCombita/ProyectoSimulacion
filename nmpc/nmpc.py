#Nonlinear Model-Predictive Control

from utils.multi_robot_plot import pintarRobotYObstaculos
from utils.create_obstacles import CrearObstaculos
import numpy as np
from scipy.optimize import minimize, Bounds

SIM_TIME = 8.
TIMESTEP = 0.1
NUMBER_OF_TIMESTEPS = int(SIM_TIME/TIMESTEP)
ROBOT_TAMANIO = 0.5
VMAX = 2
VMIN = 0.2

# Parametros para el costo de colicion
Qc = 5.
kappa = 4.

# Parametros
HORIZON_LENGTH = int(4)
NMPC_TIMESTEP = 0.3
upper_bound = [(1/np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2
lower_bound = [-(1/np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2

"""Se empieza a simular"""
def simulate():
    obstacles = CrearObstaculos(SIM_TIME, NUMBER_OF_TIMESTEPS)

    inicioParticula = np.array([5, 4]) 
    posicionFinal = np.array([5, 6])

    estadoRobot = inicioParticula
    historialEstadoRobot = np.empty((4, NUMBER_OF_TIMESTEPS))

    for i in range(NUMBER_OF_TIMESTEPS):
        # Se predice la posicion de los obstaculos en el futuro
        obstacle_predictions = PredecirPosicionObstaculo(obstacles[:, i, :])
        xref = CalcularPosicionReferencia(estadoRobot, posicionFinal,
                            HORIZON_LENGTH, NMPC_TIMESTEP)
        
        # Calcular la velosidad
        vel, velocity_profile = CalcularVelocidad(
            estadoRobot, obstacle_predictions, xref)
        estadoRobot = ActualizarEstado(estadoRobot, vel, TIMESTEP)
        historialEstadoRobot[:2, i] = estadoRobot

    pintarRobotYObstaculos(
        historialEstadoRobot, obstacles, ROBOT_TAMANIO, NUMBER_OF_TIMESTEPS, SIM_TIME)


def CalcularVelocidad(estadoRobot, obstacle_predictions, xref):
    u0 = np.random.rand(2*HORIZON_LENGTH)
    def cost_fn(u): return CostoTotal(
        u, estadoRobot, obstacle_predictions, xref)

    bounds = Bounds(lower_bound, upper_bound)

    res = minimize(cost_fn, u0, method='SLSQP', bounds=bounds)
    velocity = res.x[:2]
    return velocity, res.x

#Verifica de donde parte y a donde tiene que llegar y que tiempo tiene por paso
#y con esto normaliza la linea generada, si es menor a 0.1 eso quiere decir que se va aquedar quieto
#si no, se calcula el nuevo array de movimiento y lo multiplica por su velocidad máxima, para despues
#crear el array con el camino de referencia
def CalcularPosicionReferencia(inicio, llegada, number_of_steps, timestep):
    dir_vector = (llegada - inicio)
    norm = np.linalg.norm(dir_vector)
    if norm < 0.1:
        nueva_final = inicio
    else:
        dir_vector = dir_vector / norm
        nueva_final = inicio + dir_vector * VMAX * timestep * number_of_steps
    return np.linspace(inicio, nueva_final, number_of_steps).reshape((2*number_of_steps))


def CostoTotal(u, robot_state, obstacle_predictions, xref):
    x_robot = ActualizarEstado(robot_state, u, NMPC_TIMESTEP)
    c1 = trackingCosto(x_robot, xref)
    c2 = CostoTotalChoque(x_robot, obstacle_predictions)
    total = c1 + c2
    return total


def trackingCosto(x, xref):
    return np.linalg.norm(x-xref)


def CostoTotalChoque(robot, obstacles):
    total_cost = 0
    for i in range(HORIZON_LENGTH):
        for j in range(len(obstacles)):
            obstacle = obstacles[j]
            rob = robot[2 * i: 2 * i + 2]
            obs = obstacle[2 * i: 2 * i + 2]
            total_cost += CostoChoque(rob, obs)
    return total_cost


def CostoChoque(x0, x1):
    d = np.linalg.norm(x0 - x1)
    cost = Qc / (1 + np.exp(kappa * (d - 2* ROBOT_TAMANIO)))
    return cost


def PredecirPosicionObstaculo(obstaculo):
    obstacle_predictions = []
    #recorre sobre el numero de filas de la lista de obstaculos
    for i in range(np.shape(obstaculo)[1]):
        obstaculo2 = obstaculo[:, i]
        positionObstaculo = obstaculo2[:2]
        valocidadObstaculo = obstaculo2[2:]
        #Se calcula el control futuro del obstáculo multiplicando su velocidad actual por una matriz de identidad apilada verticalmente 
        controlFuturo = np.vstack([np.eye(2)] * HORIZON_LENGTH) @ valocidadObstaculo
        prediccionObstaculo = ActualizarEstado(positionObstaculo, controlFuturo, NMPC_TIMESTEP)
        obstacle_predictions.append(prediccionObstaculo)
    return obstacle_predictions

#Se calcula el estado del sistema aplicando la secuencia de control para no chocar
def ActualizarEstado(posicionObstaculo, controlFuturo, timestep):
    N = int(len(controlFuturo) / 2)
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(2))

    nuevoEstado = np.vstack([np.eye(2)] * int(N)) @ posicionObstaculo + kron @ controlFuturo * timestep

    return nuevoEstado
