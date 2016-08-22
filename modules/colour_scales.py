__author__ = 'Daniel'

import matplotlib.colors as colors
def rainbow_bright():

    cdict = {
        'red':   [(0.00,  0.50, 0.50),
                  (0.02,  0.50, 0.50),
                  (0.05,  0.50, 0.50),
                  (0.10,  0.50, 0.50),
                  (0.15,  0.50, 0.50),
                  (0.20,  0.75, 0.75),
                  (0.30,  1.00, 1.00),
                  (0.40,  1.00, 1.00),
                  (0.50,  1.00, 1.00),
                  (0.60,  1.00, 1.00),
                  (0.75,  1.00, 1.00),
                  (1.00,  0.75, 0.75)],

        'green': [(0.00,  0.50, 0.50),
                  (0.02,  0.75, 0.75),
                  (0.05,  1.00, 1.00),
                  (0.10,  1.00, 1.00),
                  (0.15,  1.00, 1.00),
                  (0.20,  1.00, 1.00),
                  (0.30,  1.00, 1.00),
                  (0.40,  0.75, 0.75),
                  (0.50,  0.50, 0.50),
                  (0.60,  0.50, 0.50),
                  (0.75,  0.50, 0.50),
                  (1.00,  0.50, 0.50)],

        'blue':  [(0.00,  1.00, 1.00),
                  (0.02,  1.00, 1.00),
                  (0.05,  1.00, 1.00),
                  (0.10,  0.75, 0.75),
                  (0.15,  0.50, 0.50),
                  (0.20,  0.50, 0.50),
                  (0.30,  0.50, 0.50),
                  (0.40,  0.50, 0.50),
                  (0.50,  0.50, 0.50),
                  (0.60,  0.75, 0.75),
                  (0.75,  1.00, 1.00),
                  (1.00,  1.00, 1.00)]}

    return colors.LinearSegmentedColormap('RainbowBright', cdict)


def rainbow_king():

    cdict = {
        'red':   [(0.00,  0.05, 0.05),
                  (0.02,  0.08, 0.08),
                  (0.05,  0.09, 0.09),
                  (0.10,  0.30, 0.30),
                  (0.15,  0.63, 0.63),
                  (0.20,  1.00, 1.00),
                  (0.30,  1.00, 1.00),
                  (0.40,  1.00, 1.00),
                  (0.50,  1.00, 1.00),
                  (0.60,  1.00, 1.00),
                  (0.75,  0.95, 0.95),
                  (1.00,  0.51, 0.51)],

        'green': [(0.00,  0.00, 0.00),
                  (0.02,  0.00, 0.00),
                  (0.05,  0.32, 0.32),
                  (0.10,  0.58, 0.58),
                  (0.15,  0.81, 0.81),
                  (0.20,  1.00, 1.00),
                  (0.30,  0.65, 0.65),
                  (0.40,  0.50, 0.50),
                  (0.50,  0.38, 0.38),
                  (0.60,  0.00, 0.00),
                  (0.75,  0.00, 0.00),
                  (1.00,  0.00, 0.00)],

        'blue':  [(0.00,  0.31, 0.31),
                  (0.02,  0.49, 0.49),
                  (0.05,  0.42, 0.42),
                  (0.10,  0.30, 0.30),
                  (0.15,  0.12, 0.12),
                  (0.20,  0.00, 0.00),
                  (0.30,  0.05, 0.05),
                  (0.40,  0.00, 0.00),
                  (0.50,  0.05, 0.05),
                  (0.60,  0.00, 0.00),
                  (0.75,  0.51, 0.51),
                  (1.00,  0.25, 0.25)]}

    return colors.LinearSegmentedColormap('RainbowKing', cdict)


def moores_seven_owt():

    cdict = {
        'red':   [(0.00,  0.00, 0.40),
                  (0.14,  0.40, 0.20),
                  (0.29,  0.20, 0.60),
                  (0.43,  0.60, 0.60),
                  (0.57,  0.60, 0.40),
                  (0.71,  0.40, 0.80),
                  (0.86,  0.80, 0.60),
                  (1.00,  0.60, 0.00)],

        'green': [(0.00,  0.00, 0.20),
                  (0.14,  0.20, 0.40),
                  (0.29,  0.40, 1.00),
                  (0.43,  1.00, 0.80),
                  (0.57,  0.80, 0.60),
                  (0.71,  0.60, 0.60),
                  (0.86,  0.60, 0.40),
                  (1.00,  0.40, 0.00)],

        'blue':  [(0.00,  0.00, 0.60),
                  (0.14,  0.60, 0.80),
                  (0.29,  0.80, 1.00),
                  (0.43,  1.00, 0.20),
                  (0.57,  0.20, 0.40),
                  (0.71,  0.40, 0.40),
                  (0.86,  0.40, 0.00),
                  (1.00,  0.00, 0.00)]}

    return colors.LinearSegmentedColormap('MooresSevenOWT', cdict)


def cyano_portion():

    cdict = {
        'red':   [(0.00,  0.17, 0.17),
                  (0.25,  0.17, 0.17),
                  (0.50,  0.17, 0.17),
                  (0.75,  0.17, 0.17),
                  (1.00,  0.17, 0.17)],

        'green': [(0.00,  0.40, 0.40),
                  (0.25,  0.72, 0.72),
                  (0.50,  0.72, 0.72),
                  (0.75,  0.67, 0.67),
                  (1.00,  0.34, 0.34)],

        'blue':  [(0.00,  0.17, 0.17),
                  (0.25,  0.43, 0.43),
                  (0.50,  0.71, 0.71),
                  (0.75,  0.95, 0.95),
                  (1.00,  0.66, 0.66)]}

    return colors.LinearSegmentedColormap('CyanoPortion', cdict)


def floating_portion():

    cdict = {
        'red':   [(0.00,  0.17, 0.17),
                  (0.25,  0.43, 0.43),
                  (0.50,  0.71, 0.71),
                  (0.75,  0.95, 0.95),
                  (1.00,  0.66, 0.66)],

        'green': [(0.00,  0.40, 0.40),
                  (0.25,  0.72, 0.72),
                  (0.50,  0.72, 0.72),
                  (0.75,  0.67, 0.67),
                  (1.00,  0.34, 0.34)],

        'blue':  [(0.00,  0.17, 0.17),
                  (0.25,  0.27, 0.27),
                  (0.50,  0.53, 0.53),
                  (0.75,  0.95, 0.95),
                  (1.00,  0.71, 0.71)]}

    return colors.LinearSegmentedColormap('FloatingPortion', cdict)


def num_obs_scale():

    cdict = {
        'red':   [(0.000,  0.00, 0.5),
                  (0.167,  0.5, 0.5),
                  (0.333,  0.5, 0.75),
                  (0.500,  0.75, 1.0),
                  (0.667,  1.0, 1.0),
                  (0.833,  1.0, 1.0),
                  (1.000,  1.0, 0.00)],

        'green': [(0.000,  0.00, 0.5),
                  (0.167,  0.5, 1.0),
                  (0.333,  1.0, 1.0),
                  (0.500,  1.0, 1.0),
                  (0.667,  1.0, 0.75),
                  (0.833,  0.75, 0.5),
                  (1.000,  0.5, 0.00)],

        'blue':  [(0.000,  0.00, 1.0),
                  (0.167,  1.0, 1.0),
                  (0.333,  1.0, 0.75),
                  (0.500,  0.75, 0.5),
                  (0.667,  0.5, 0.5),
                  (0.833,  0.5, 0.5),
                  (1.000,  0.5, 0.00)]}

    return colors.LinearSegmentedColormap('NumObsScale', cdict)


def extent_true():

    cdict = {
        'red':   [(0.00,  1.00, 1.00),
                  (0.33,  1.00, 0.30),
                  (0.66,  0.30, 1.00),
                  (1.00,  1.00, 1.00)],

        'green': [(0.00,  1.00, 1.00),
                  (0.33,  1.00, 0.30),
                  (0.66,  0.30, 1.00),
                  (1.00,  1.00, 1.00)],

        'blue':  [(0.00,  1.00, 1.00),
                  (0.33,  1.00, 0.30),
                  (0.66,  0.30, 0.30),
                  (1.00,  0.30, 0.30)]}

    return colors.LinearSegmentedColormap('ExtentTrue', cdict)