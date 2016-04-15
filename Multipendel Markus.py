__author__ = 'philipp'

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib
from threading import Thread
from multiprocessing import Queue, Process, Pool
import time
import threading

matplotlib.interactive(True)
iterations = 1

class Drawer(Thread):

    def __init__(self, q):
        Thread.__init__(self)
        plot1 = plt.subplot()
        plot1.set_xlim([-SCREENSIZE/2, SCREENSIZE/2])
        plot1.set_ylim([-SCREENSIZE/2, SCREENSIZE/2])
        self.data, = plot1.plot(x, y, '-o')
        self.data.set_xdata(np.insert(0, 0, 0))
        self.data.set_ydata(np.insert(0, 0, 0))
        self.q = q
        plt.show()

    def run(self):
        DT = dt*iterations
        t = time.time()
        while True:
            if time.time() - t > DT:
                x, x0, y, y0 = self.q.get()
                self.data.set_xdata(np.insert(x, 0, x0))
                self.data.set_ydata(np.insert(y, 0, y0))
                plt.draw()
                
            #time.sleep(0.01)

def update_positions():
    global phi, phi_dot, phi_ddot, dt, l, N
    global x, x0, y, y0
    hx = 0
    hy = 0
    A = np.multiply(np.outer(np.cos(phi), np.cos(phi)) + np.outer(np.sin(phi), np.sin(phi)), M)
    D = -c * np.sin(phi)
    for i in xrange(N):
        D[i] *= (N - i)
    B = np.multiply(np.outer(np.sin(phi), np.cos(phi)) - np.outer(np.cos(phi), np.sin(phi)), M)
    D -= np.dot(B, phi_dot ** 2)
    phi_ddot = np.dot(lin.inv(A), D) - gamma * phi_dot
    phi_dot += phi_ddot * dt
    phi = phi_dot * dt + phi
    for j in xrange(N):
        hx += np.sin(phi[j])
        x[j] = x0 + l * hx
        hy += np.cos(phi[j])
        y[j] = y0 - l * hy
    #q.put([x, x0, y, y0])
    return x, x0, y, y0


def Stufenmatrix(N):
    B = np.zeros([N, N])

    for i in xrange(N):
        for j in xrange(N - 1, i - 1, -1):
            B[j][i] = N - j
            if j == i:
                for k in xrange(j):
                    B[k][j] = N - j
    return B


N = 3  # Anzahl der Pendel
l = 1# Laenge der Pendel
dt = 0.009  # Zeitschritt
SCREENSIZE = l*2*N
gamma = 0 # Daempfung
g = 1  # Erdbeschleunigung
c = g / l
x0 = 0
y0 = 0
phi = 6.28 * np.random.random_sample((N,))  # Anfangswinkel
phi_dot = 0 * np.random.random_sample((N,))  # Anfangs(winkel)geschwindigkeiten
phi_ddot = np.zeros(N)  # Anfangs(winkel)beschleunigungen
A = np.zeros([N, N])
D = np.zeros(N)  # A,D,M Hilfsmatrizen
M = Stufenmatrix(N)
x = np.zeros(N)  # x,y: kartesische Koordinaten
y = np.zeros(N)

def calcSchleife():
    t = time.time()
    while True:
        #update_positions(q)
        if time.time() - t > dt:
            yield update_positions()
        #return phi, phi_dot, phi_ddot, dt, l, N



def main():
    try:
        global phi, phi_dot, phi_ddot, dt, l, N
        global x, x0, y, y0
        i = 0

        # plot1 = plt.subplot()
        # plot1.set_xlim([-SCREENSIZE/2, SCREENSIZE/2])
        # plot1.set_ylim([-SCREENSIZE/2, SCREENSIZE/2])
        # data, = plot1.plot(x, y, '-')

        q = Queue()

        #p = Process(target=calcSchleife, args=(q))
        #p.start()

        d = Drawer(q)
        d.start()


        #while True:
            # i += 1
            #x, x0, y, y0 = q.get()
        for a, b, c, d in calcSchleife():
            # data.set_xdata(np.insert(a, 0, b))
            # data.set_ydata(np.insert(c, 0, d))
            #threading._start_new_thread(plt.draw,())
            q.put([a,b,c,d],block=False)
            for i in range(iterations):
                calcSchleife().next()

            #time.sleep(0.0001)



    except KeyboardInterrupt:
        exit()


if __name__ == '__main__':
    main()