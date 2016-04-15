#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A simulation of a multipendulum"""

__author__ = 'Philipp Rosenberger, Tobias Obermayer, Markus Weber'

# from __builtin__ import range  # forward-compatible (xrange=range) faster?

import sys
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from help_functions import *

zeitraffer = 40
# TODO: zeitraffer, overflow


class Multipendulum(object):
    """A simulation of a multipendulum"""
    def __init__(self, length_pendulum=1, number_pendulums=3,
                 damping=0., gravitational_acceleration=1.,
                 fps=25, length_trace=100):

        self.length_pendulum = length_pendulum
        self.number_pendulums = number_pendulums
        self.damping = damping
        self.gravitational_acceleration = gravitational_acceleration
        self.fps = fps
        self.time_step = 1.0/fps
        self.length_trace = length_trace

        n = self.number_pendulums
        self.__c = self.gravitational_acceleration / self.length_pendulum
        self.x0 = 0  # Startposition
        self.y0 = 0
        self.phi = np.array([])
        self.random_angles()
        self.phi_dot = 0 * np.random.random_sample((n,))  # Anfangs(winkel)geschwindigkeiten
        self.phi_ddot = np.zeros(n)  # Anfangs(winkel)beschleunigungen
        self.__A = np.zeros([n, n])
        self.__D = np.zeros(n)  # A,D,M Hilfsmatrizen
        self.__M = self.__stage_matrix(n)
        self.x = np.zeros(n)  # x,y: kartesische Koordinaten
        self.y = np.zeros(n)
        self.trace_data_x = np.array([])
        self.trace_data_y = np.array([])

        self.__fig = plt.figure()
        size = self.length_pendulum * self.number_pendulums
        self.__plot1 = self.__fig.add_subplot(111, autoscale_on=False, xlim=(-size, size), ylim=(-size, size))
        self.__plot1.grid()

        self.__data, = self.__plot1.plot(self.x, self.y, '-o', linewidth=2)
        self.__trace, = self.__plot1.plot(self.x, self.y, '-', linewidth=1)

        # self.start_animation()

    def almost_vertical_angles(self):
        angles = [np.pi-0.05+0.1*np.random.random() for _ in range(self.number_pendulums)]
        self.phi = np.array(angles)

    def random_angles(self):
        self.phi = 2 * np.pi * np.random.random_sample((self.number_pendulums,))

    def update_positions(self):
        phi = self.phi
        t = self.time_step
        l = self.length_pendulum
        n = self.number_pendulums

        hx = 0
        hy = 0
        __A = np.multiply(np.outer(np.cos(phi), np.cos(phi)) + np.outer(np.sin(phi), np.sin(phi)), self.__M)
        __D = np.sin(phi) * -self.__c
        for i in range(n):
            __D[i] *= (n - i)
        __B = np.multiply(np.outer(np.sin(phi), np.cos(phi)) - np.outer(np.cos(phi), np.sin(phi)), self.__M)
        __D -= np.dot(__B, self.phi_dot ** 2)
        self.phi_ddot = np.dot(lin.inv(__A), __D) - self.damping * self.phi_dot
        self.phi_dot += self.phi_ddot * t
        self.phi = self.phi_dot * t + phi
        for j in range(n):
            hx += np.sin(phi[j])
            self.x[j] = self.x0 + l * hx
            hy += np.cos(phi[j])
            self.y[j] = self.y0 - l * hy

    @staticmethod
    def __stage_matrix(size):
        matrix = np.zeros([size, size])

        for i in range(size):
            for j in range(size - 1, i - 1, -1):
                matrix[j][i] = size - j
                if j == i:
                    for k in range(j):
                        matrix[k][j] = size - j
        return matrix

    def start_animation(self, infinity_loop=True):
        ani = animation.FuncAnimation(self.__fig, self.__animate, 1000,
                                      interval=self.time_step/zeitraffer,
                                      blit=True, repeat=infinity_loop,
                                      init_func=self.__init)
        try:
            plt.show()
        except AttributeError:
            pass

    def __init(self):
        self.__data.set_data([], [])
        self.__trace.set_data([], [])
        self.update_positions()
        return self.__data, self.__trace

    def __animate(self, _):
        if len(self.trace_data_x) > self.length_trace:
            self.trace_data_x = np.delete(self.trace_data_x, 0)
            self.trace_data_y = np.delete(self.trace_data_y, 0)

        self.trace_data_x = np.append(self.trace_data_x, self.x[-1])
        self.trace_data_y = np.append(self.trace_data_y, self.y[-1])

        self.__data.set_data(np.insert(self.x, 0, self.x0), np.insert(self.y, 0, self.y0))
        self.__trace.set_data(self.trace_data_x, self.trace_data_y)
        self.update_positions()
        return self.__data, self.__trace

    def __trace(self):
        pass


def main(argv):
    count = 3
    try:
        count = int(argv[0])
    except:
        pass

    damp = 0.0
    try:
        damp = float(argv[1])
    except:
        pass

    pendulum = Multipendulum(number_pendulums=count, damping=damp)
    pendulum.length_trace = 6000
    pendulum.almost_vertical_angles()
    pendulum.start_animation(infinity_loop=True)

if __name__ == "__main__":
    main(sys.argv[1:])