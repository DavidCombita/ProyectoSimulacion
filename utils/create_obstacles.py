import numpy as np

def CrearObstaculos(sim_time, num_timesteps):
    # Obstaculo 1, 90 grados
    v = -2
    posicionArranque = np.array([5, 12])
    obst = CrearRobot(posicionArranque, v, np.pi/2, sim_time,
                        num_timesteps).reshape(4, num_timesteps, 1)
    obstacles = obst
    # Obstaculo 2, 0 grados
    v = 2
    posicionArranque = np.array([0, 5])
    obst = CrearRobot(posicionArranque, v, 0, sim_time, num_timesteps).reshape(
        4, num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))
    # Obstaculo 3, 135 grados hacia abajo
    v = 2
    posicionArranque = np.array([10, 10])
    obst = CrearRobot(posicionArranque, v, -np.pi * 3 / 4, sim_time, num_timesteps).reshape(4,
                                                                                num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))
    # Obstaculo 4, 135 grados hacia arriba
    v = 2
    posicionArranque = np.array([7.5, 2.5])
    obst = CrearRobot(posicionArranque, v, np.pi * 3 / 4, sim_time, num_timesteps).reshape(4,
                                                                               num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))

    #se retorna la lista de obstaculos que creamos
    return obstacles


def CrearRobot(posicionArranque, velocidadSalida, anguloSalida, sim_time, num_timesteps):
    # Crea obstaculos que comienzan en posicionArranque y se van moviendo con velocidadSalida 
    # y en la direccion del angulo de Salida

    #Se genera un linea un números espaciados uniformes
    linea = np.linspace(0, sim_time, num_timesteps)
    anguloSalida = anguloSalida * np.ones(np.shape(linea))
    
    vx = velocidadSalida * np.cos(anguloSalida)
    vy = velocidadSalida * np.sin(anguloSalida)
    velocidadSalida = np.stack([vx, vy])

    #Se termina de organizar el array que tenemos de velocidad salida, y luego lo concatenamos para 
    #tener el vector de salida del obstaculo
    posicionArranque = posicionArranque.reshape((2, 1))
    p = posicionArranque + np.cumsum(velocidadSalida, axis=1) * (sim_time / num_timesteps)
    p = np.concatenate((p, velocidadSalida))
    return p
