# -*- coding: utf-8 -*-
# Final.py
# Python 3.7
"""
Author: Jake Gurevitch & Brian Song
Created: 4/29/20

Description
    Simulation of Water Molecules for PHYS260 Final Project
-----------
"""
import numpy as np
from random import random
import vpython as vp

L = 1e-10 # lengthscale (m)

# ================================================================================
# Scene set up
# ================================================================================
w = 50  # box width (m)
backgroundColor = vp.color.cyan
wallColor = vp.color.gray(luminance=0.9)
scene = vp.canvas(width=10 * w, height=10 * w, center=vp.vector(w / 2, w / 2, 0), range=0.6 * w, fov=0.01,
                  background=backgroundColor,
                  userzoom=False, autoscale=False, userspin=False)
# box
vp.box(pos=vp.vector(0, w / 2, 0), size=vp.vector(1, w, 1), color=wallColor)
vp.box(pos=vp.vector(w, w / 2, 0), size=vp.vector(1, w, 1), color=wallColor)
vp.box(pos=vp.vector(w / 2, w, 0), size=vp.vector(w, 1, 1), color=wallColor)
vp.box(pos=vp.vector(w / 2, 0, 0), size=vp.vector(w, 1, 1), color=wallColor)

# ================================================================================
# Calculate the positions of the two hydrogen atoms given the position and
# orientation of the water molecule
# ================================================================================

# oDir is the angle that the water molecule makes counterclockwise
# relative to the horizontal (like a unit circle)
def calculateHPositions(oX, oY, oDir):

    def calculatePos(angle):
        xPos = oX + bondLength * np.cos(angle)
        yPos = oY + bondLength * np.sin(angle)
        return (xPos, yPos)

    def calculateHLeftPos():
        angle = oDir - (bondAngle / 2)
        return calculatePos(angle)

    def calculateHRightPos():
        angle = oDir + (bondAngle / 2)
        return calculatePos(angle)

    h1X, h1Y = calculateHLeftPos()
    h2X, h2Y = calculateHRightPos()

    return (h1X, h1Y, h2X, h2Y)

# ================================================================================
# Computes the spring force from the wall at a given position
# ================================================================================

def computeWallForce(x, y):
    Fx = 0
    Fy = 0

    if x < wallProximity:
        Fx = kWall * (wallProximity - x)
    elif x > w * L - wallProximity:
        Fx = kWall * (w * L - wallProximity - x)

    if y < wallProximity:
        Fy = kWall * (wallProximity - y)
    elif y > w * L - wallProximity:
        Fy = kWall * (w * L - wallProximity - y)

    return (Fx, Fy)

# ================================================================================
# Computes bond force (as a spring force) between two atoms (an O and an H, in
# this case)
# returns the force (experienced by the H) as its x and y components, as well
# as the angle between the O and the H
# ================================================================================

def computeBondSpringForce(oX, oY, hX, hY):
    distToH = np.sqrt((oX - hX)**2 + (oY - hY)**2)
    FonH = -kR * (distToH - bondLength)

    yDist = hY - oY
    xDist = hX - oX

    relativeAngle = np.arctan(yDist / xDist)
    absoluteAngle = relativeAngle

    # right side of unit circle
    if xDist >= 0:
        # quadrant IV
        if yDist < 0:
            absoluteAngle = 2 * np.pi + relativeAngle
    # left side
    else:
        absoluteAngle = np.pi + relativeAngle

    Fx = FonH * np.cos(absoluteAngle)
    Fy = FonH * np.sin(absoluteAngle)

    return (Fx, Fy, absoluteAngle)

# ================================================================================
# Computes Coulombic force between two charged atoms
# force is from atom 2 on atom 1
# ================================================================================
def computeCoulombicForce(x1, y1, x2, y2, q1, q2):
    d12 = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

    F = -kE * q1 * q2 / d12**2

    Fx = F * (x2 - x1) / d12
    Fy = F * (y2 - y1) / d12

    return (Fx, Fy)

# ================================================================================
# Compute forces and potential energy
# ================================================================================

def force(oX, oY, hX, hY):

    FxO = np.zeros(N, float) # x component of net force on each oxygen
    FyO = np.zeros(N, float) # y component of net force on each oxygen
    FxH = np.zeros(2 * N, float) # x component of net force on each hydrogen
    FyH = np.zeros(2 * N, float) # y component of net force on each hydrogen

    # Calculates Fx and Fy for Os and Hs
    for i in range(N):
        h1Pos = 2 * i
        h2Pos = 2 * i + 1

        x = oX[i]
        y = oY[i]
        h1X = hX[h1Pos]
        h1Y = hY[h1Pos]
        h2X = hX[h2Pos]
        h2Y = hY[h2Pos]

        # collision with walls
        oWallForce = computeWallForce(x, y)

        FxO[i] += oWallForce[0]
        FyO[i] += oWallForce[1]

        # calculating spring force for covalent bonds
        FxOnH1, FyOnH1, h1Angle = computeBondSpringForce(x, y, h1X, h1Y)
        FxOnH2, FyOnH2, h2Angle = computeBondSpringForce(x, y, h2X, h2Y)

        FxO[i] -= (FxOnH1 + FxOnH2)
        FyO[i] -= (FyOnH1 + FyOnH2)
        FxH[h1Pos] += FxOnH1
        FyH[h1Pos] += FyOnH1
        FxH[h2Pos] += FxOnH2
        FyH[h2Pos] += FyOnH2

        # calculating restorative angle force
        magAngleDifference = np.abs(h2Angle - h1Angle)
        if magAngleDifference > np.pi:
            magAngleDifference = 2 * np.pi - magAngleDifference

        torque = -kTheta * (magAngleDifference - bondAngle)

        F1 = np.abs(torque / np.sqrt((h1X - x) ** 2 + (h1Y - y) ** 2))
        F2 = np.abs(torque / np.sqrt((h2X - x) ** 2 + (h2Y - y) ** 2))

        forceAngle1 = h1Angle
        forceAngle2 = h2Angle
        angleDifference = h1Angle - h2Angle

        # want to get closer
        if torque <= 0:
            # decrease h1 angle to get closer
            if angleDifference <= -np.pi or (angleDifference >= 0 and angleDifference <= np.pi):
                forceAngle1 = h1Angle - np.pi/2
                forceAngle2 = h2Angle + np.pi/2
            else:
                forceAngle1 = h1Angle + np.pi/2
                forceAngle2 = h2Angle - np.pi/2
        # want to get further
        elif torque > 0:
            # decrease h1 angle to get further away
            if angleDifference <= -np.pi or (angleDifference >= 0 and angleDifference <= np.pi):
                forceAngle1 = h1Angle + np.pi/2
                forceAngle2 = h2Angle - np.pi/2
            else:
                forceAngle1 = h1Angle - np.pi/2
                forceAngle2 = h2Angle + np.pi/2

        F1x = F1 * np.cos(forceAngle1)
        F1y = F1 * np.sin(forceAngle1)
        F2x = F2 * np.cos(forceAngle2)
        F2y = F2 * np.sin(forceAngle2)

        FxH[h1Pos] += F1x
        FyH[h1Pos] += F1y
        FxH[h2Pos] += F2x
        FyH[h2Pos] += F2y

        # Lennard-Jones interaction between the different molecules
        for j in range(i + 1, N):
            deltax = oX[j] - x
            deltay = oY[j] - y
            Rij = np.sqrt(deltax * deltax + deltay * deltay)

            # if molecules within cutoff, compute LJ force
            if Rij <= cutoff:
                # force
                Fijx = (-12 * ljTwelfth / Rij ** 14 + 6 * ljSixth / Rij ** 8) * deltax
                Fijy = (-12 * ljTwelfth / Rij ** 14 + 6 * ljSixth / Rij ** 8) * deltay

                # net forces
                FxO[i] += Fijx
                FyO[i] += Fijy
                FxO[j] += -Fijx
                FyO[j] += -Fijy

        # Calculating Coulomb force
        for j in range(i + 1, N):
            # pulling out jth water molecule
            h3Pos = 2 * j
            h4Pos = 2 * j + 1

            x2 = oX[j]
            y2 = oY[j]
            h3X = hX[h3Pos]
            h3Y = hY[h3Pos]
            h4X = hX[h4Pos]
            h4Y = hY[h4Pos]

            # calculating forces
            FOOx, FOOy = computeCoulombicForce(x, y, x2, y2, oCharge, oCharge) # force between the oxygens
            FOH3x, FOH3y = computeCoulombicForce(x, y, h3X, h3Y, oCharge, hCharge) # force on o1 from h3
            FOH4x, FOH4y = computeCoulombicForce(x, y, h4X, h4Y, oCharge, hCharge) # force on o1 from h4
            FH1Ox, FH1Oy = computeCoulombicForce(h1X, h1Y, x2, y2, hCharge, oCharge) # force on h1 from o2
            FH1H3x, FH1H3y = computeCoulombicForce(h1X, h1Y, h3X, h3Y, hCharge, hCharge) # force on h1 from h3
            FH1H4x, FH1H4y = computeCoulombicForce(h1X, h1Y, h4X, h4Y, hCharge, hCharge) # force on h1 from h4
            FH2Ox, FH2Oy = computeCoulombicForce(h2X, h2Y, x2, y2, hCharge, oCharge) # force on h2 from o2
            FH2H3x, FH2H3y = computeCoulombicForce(h2X, h2Y, h3X, h3Y, hCharge, hCharge) # force on h2 from h3
            FH2H4x, FH2H4y = computeCoulombicForce(h2X, h2Y, h4X, h4Y, hCharge, hCharge) # force on h2 from h4

            # forces on ith water
            FxO[i] += (FOOx + FOH3x + FOH4x)
            FyO[i] += (FOOy + FOH3y + FOH4y)
            FxH[h1Pos] += (FH1Ox + FH1H3x + FH1H4x)
            FyH[h1Pos] += (FH1Oy + FH1H3y + FH1H4y)
            FxH[h2Pos] += (FH2Ox + FH2H3x + FH2H4x)
            FyH[h2Pos] += (FH2Oy + FH2H3y + FH2H4y)

            # forces on jth water
            FxO[j] -= (FOOx + FH1Ox + FH2Ox)
            FyO[j] -= (FOOy + FH1Oy + FH2Oy)
            FxH[h3Pos] -= (FOH3x + FH1H3x + FH2H3x)
            FyH[h3Pos] -= (FOH3y + FH1H3y + FH2H3y)
            FxH[h4Pos] -= (FOH4x + FH1H4x + FH2H4x)
            FxH[h4Pos] -= (FOH4y + FH1H4y + FH2H4y)

    return FxO, FyO, FxH, FyH

# ================================================================================
# Simulation parameters
# ================================================================================
# Atomic Properties (using TIP3P model)
oMass = 16 * (1.66054e-27 / 1) # kg
hMass = 1.01 * (1.66054e-27 / 1) # kg
oRad = 1.52 * (1 / 1e10) # m
hRad = 1.2 * (1 / 1e10) # m
bondAngle = 104.52 * (np.pi / 180) # rad
bondLength = 0.9572 * (1 / 1e10) # m
kR = 6.5 * (1.602e-19) * (1e10)**2 # OH bond stiffness (J/m^2)
kTheta = 1 * 1.602e-19 # Bond angle stiffness (J/radian^2)
sigma = 3.15061 * (1 / 1e10) # LJ Radius (m)
epsilon = 0.6364 * 1000 *  (1 / 6.022e23)  # J/atom
hCharge = +0.4170 * (1.602e-19 / 1) # C
oCharge = -0.8340 * (1.602e-19 / 1) # C
kE = 8.988e9 # constant for couloumbic force (Jm/C^2)

# General Parameters
N = 36
dt = 1e-15 # timestep (s)
kWall = 600 # stiffness of walls (J/m^2)
nstemp = 50 # number of time steps before velocity rescaling
T0 = 200 # point temperature (K)
wallProximity = 2**(1/6)*sigma/2 # how close atoms can be to the wall before being bounced back

# Values for calculating LJ Potential
sigmaSixth = sigma**6                 # sigma^6
sigmaTwelfth = sigmaSixth**2          # sigma^12
ljSixth = 4*epsilon*sigmaSixth        # 4*epsilon*sigma**6
ljTwelfth = 4*epsilon*sigmaTwelfth    # 4*epsilon*sigma**12
cutoff = 3*sigma                      # cutoff distance for LJ force calculation

# Parameters for Drawing
oColor = vp.color.red
hColor = vp.color.white
padding = 2 * (oRad/L)

# ================================================================================
# Initialization
# ================================================================================

# arrays for position: there are two different sets of arrays, one for oxygens
# and one for hydrogens. o[i]'s Hydrogen's are at h[2*i] and h[2*i + 1]
oX = np.zeros(N, float)
oY = np.zeros(N, float)
hX = np.zeros(2 * N, float)
hY = np.zeros(2 * N, float)

# arrays for velocity
vxO = np.array([1500 * (0.5 - random()) for i in range(N)], float)
vyO = np.array([1500 * (0.5 - random()) for i in range(N)], float)
vxmidO = np.zeros(N, float)
vymidO = np.zeros(N, float)
# instantiating the hydrogens with the same speed as their water
vxH = np.array([vxO[int(i / 2)] for i in range(2 * N)], float)
vyH = np.array([vyO[int(i / 2)] for i in range(2 * N)], float)
vxmidH = np.zeros(2 * N, float)
vymidH = np.zeros(2 * N, float)

t = 0
counter = 0

# lists of atoms (for redrawing)
os = []
hs = []

# parameters for spacing molecules
numRows = int(np.ceil(np.sqrt(N)))
moleculesPerRow = int(np.ceil(N / numRows))
dx = 1 if moleculesPerRow <= 1 else (w - 2 * padding)/(moleculesPerRow - 1)
dy = 1 if numRows <= 1 else (w - 2 * padding)/(numRows - 1)

# initializing molecules
for i in range(numRows):
    for j in range(min(moleculesPerRow, N - moleculesPerRow * i)):
        # initializing oxygen atom
        oLocation = i * moleculesPerRow + j

        xCoord = j * (dx * L) + (padding * L)
        yCoord = i * (dy * L) + (padding * L)
        mcAngle = random() * 2 * np.pi

        oX[oLocation] = xCoord
        oY[oLocation] = yCoord

        # initializing accompanying hydrogens
        h1Location = 2 * oLocation
        h2Location = h1Location + 1

        h1X, h1Y, h2X, h2Y = calculateHPositions(xCoord, yCoord, mcAngle)
        hX[h1Location] = h1X
        hY[h1Location] = h1Y
        hX[h2Location] = h2X
        hY[h2Location] = h2Y

        # drawing in molecule
        os.append(vp.sphere(pos=vp.vector(xCoord / L, yCoord / L, 0), radius=oRad/L, color=oColor))
        hs.append(vp.sphere(pos=vp.vector(h1X / L, h1Y / L, 0), radius=hRad / L, color=hColor))
        hs.append(vp.sphere(pos=vp.vector(h2X / L, h2Y / L, 0), radius=hRad / L, color=hColor))

# Perform a half step
FxO, FyO, FxH, FyH = force(oX, oY, hX, hY)
vxmidO = vxO + 0.5 * dt * FxO / oMass
vymidO = vyO + 0.5 * dt * FyO / oMass
vxmidH = vxH + 0.5 * dt * FxH / hMass
vymidH = vyH + 0.5 * dt * FyH / hMass

# ================================================================================
# Main simulation loop
# ================================================================================

while True:

    vp.rate(1000 / dt)

    # Verlet algorithm
    oX += vxmidO * dt
    oY += vymidO * dt
    hX += vxmidH * dt
    hY += vymidH * dt
    FxO, FyO, FxH, FyH = force(oX, oY, hX, hY)
    vxO = vxmidO + 0.5 * FxO / oMass * dt
    vyO = vymidO + 0.5 * FyO / oMass * dt
    vxH = vxmidH + 0.5 * FxH / hMass * dt
    vyH = vymidH + 0.5 * FyH / hMass * dt
    vxmidO += FxO / oMass * dt
    vymidO += FyO / oMass * dt
    vxmidH += FxH / hMass * dt
    vymidH += FyH / hMass * dt

    K = 0.5 * (oMass*sum(vxO*vxO + vyO*vyO) + hMass*sum(vxH*vxH + vyH*vyH)) # kinetic energy
    T = (2.0 / 3.0) * K/(N * 1.381e-23) # temperature

    # update atom positions
    for i in range(N):
        os[i].pos = vp.vector(oX[i] / L, oY[i] / L, 0)
        hs[2 * i].pos = vp.vector(hX[2 * i] / L, hY[2 * i] / L, 0)
        hs[2 * i + 1].pos = vp.vector(hX[2 * i + 1] / L, hY[2 * i + 1] / L , 0)

    # Thermostat
    if counter % nstemp == 0 and counter > 0:
        if T == 0:
            vxmidO = np.array([1500 * (0.5 - random()) for i in range(N)], float)
            vymidO = np.array([1500 * (0.5 - random()) for i in range(N)], float)
            vxmidH = np.array([vxO[int(i / 2)] for i in range(2 * N)], float)
            vymidH = np.array([vyO[int(i / 2)] for i in range(2 * N)], float)
        else:
            gamma = np.sqrt(T0/T)
            vxmidO *= gamma
            vymidO *= gamma
            vxmidH *= gamma
            vymidH *= gamma

    counter += 1
    t += dt