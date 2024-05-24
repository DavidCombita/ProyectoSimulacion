import numpy as np

# calcula la velocidad deseada para que un robot se mueva desde su posición actual hacia una posición objetivo.
def CalcularVelocidadDeseada(current_pos, posicionFinal, tamanioRobot, vmax):
    disp_vec = (posicionFinal - current_pos)[:2]
    norm = np.linalg.norm(disp_vec)
    #Si la normal es menor que tamanioRobot / 5, devuelve un vector de velocidad cero, ya está cerca del objetivo
    if norm < tamanioRobot / 5:
        return np.zeros(2)
    disp_vec = disp_vec / norm
    np.shape(disp_vec)
    desired_vel = vmax * disp_vec
    return desired_vel
