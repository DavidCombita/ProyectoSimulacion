#Plotting tool for 2D multi-robot system

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np


def pintarRobotYObstaculos(robot, obstaculos, tamañoCirculo, num_steps, sim_time):
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 10), ylim=(0, 10))
    ax.set_aspect('equal')
    ax.grid()
    line, = ax.plot([], [], '--r')

    #Se crea el circulo del robot
    robot_patch = Circle((robot[0, 0], robot[1, 0]),
                         tamañoCirculo, facecolor='green', edgecolor='black')
    
    #Se recorre la lista de obstaculos y se guarda los circulos de los obstaculos
    listaObstaculos = []
    for obstacle in range(np.shape(obstaculos)[2]):
        obstacle = Circle((0, 0), tamañoCirculo,
                          facecolor='aqua', edgecolor='black')
        listaObstaculos.append(obstacle)

    def init():
        #se agregan circulos al grafico
        ax.add_patch(robot_patch)
        for obstacle in listaObstaculos:
            ax.add_patch(obstacle)
        line.set_data([], [])
        return [robot_patch] + [line] + listaObstaculos

    def animate(i):
        #se anima la linea creada para los obstaculos y el robot para cada paso
        robot_patch.center = (robot[0, i], robot[1, i])
        for j in range(len(listaObstaculos)):
            listaObstaculos[j].center = (obstaculos[0, i, j], obstaculos[1, i, j])
        line.set_data(robot[0, :i], robot[1, :i])
        return [robot_patch] + [line] + listaObstaculos

    #Se inicia pintando las posiciones en tiempo 0
    init()
    #Se calculan los pasos y se van pintando segun vayan pasando los pasos
    step = (sim_time / num_steps)
    for i in range(num_steps):
        animate(i)
        plt.pause(step)


def graficarRobot(robot, timestep, radius=1, is_obstacle=False):
    if robot is None:
        return
    center = robot[:2, timestep]
    x = center[0]
    y = center[1]
    if is_obstacle:
        circle = plt.Circle((x, y), radius, color='aqua', ec='black')
        plt.plot(robot[0, :timestep], robot[1, :timestep], '--r',)
    else:
        circle = plt.Circle((x, y), radius, color='green', ec='black')
        plt.plot(robot[0, :timestep], robot[1, :timestep], 'blue')

    plt.gcf().gca().add_artist(circle)
