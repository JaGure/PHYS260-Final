# -*- coding: utf-8 -*-
# Final.py
# Python 3.7
"""
Author: Jake Gurevitch
Created: 4/29/20
Modified: %(date)s

Description
    Simulation of Water Molecules for PHYS260 Final Project
-----------
"""
import numpy as np
from random import random
import vpython as vp

# ================================================================================
# Scene set up
# ================================================================================
w = 50  # box width (in Angstroms)
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
# relative to the horizontal (i.e. like a unit circle)

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
    elif x > w - wallProximity:
        Fx = kWall * (w - wallProximity - x)

    if y < wallProximity:
        Fy = kWall * (wallProximity - y)
    elif y > w - wallProximity:
        Fy = kWall * (w - wallProximity - y)

    return (Fx, Fy)

# ================================================================================
# Computes bond force (as a spring force) between two atoms (an O and an H, in
# this case)
# returns the force (experienced by the H) as its x and y components
# ================================================================================

def computeBondSpringForce(oX, oY, hX, hY, hAngle):
    distToH = np.sqrt((oX - hX)**2 + (oY - hY)**2)
    FonH = -kR * (distToH - bondLength)

    Fx = FonH * np.cos(hAngle)
    Fy = FonH * np.sin(hAngle)

    return (Fx, Fy)

# ================================================================================
# Compute forces and potential energy
# ================================================================================

def force(oX, oY, oDirs, hX, hY):

    FxO = np.zeros(N, float) # x component of net force on each oxygen
    FyO = np.zeros(N, float) # y component of net force on each oxygen
    FxH = np.zeros(2 * N, float) # x component of net force on each hydrogen
    FyH = np.zeros(2 * N, float) # y component of net force on each hydrogen

    # Calculates Fx, Fy for Os and Hs
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
        h1WallForce = computeWallForce(h1X, h1Y)
        h2WallForce = computeWallForce(h2X, h2Y)

        FxO[i] += oWallForce[0]
        FyO[i] += oWallForce[1]
        FxH[h1Pos] += h1WallForce[0]
        FyH[h1Pos] += h1WallForce[1]
        FxH[h2Pos] += h2WallForce[0]
        FyH[h2Pos] += h2WallForce[1]

        # calculating spring force for covalent bonds
        FxOnH1, FyOnH1 = computeBondSpringForce(x, y, h1X, h1Y, oDir[i] - bondAngle / 2)
        FxOnH2, FyOnH2 = computeBondSpringForce(x, y, h2X, h2Y, oDir[i] + bondAngle / 2)

        FxO[i] -= (FxOnH1 + FxOnH2)
        FyO[i] -= (FyOnH1 + FyOnH2)
        FxH[h1Pos] += FxOnH1
        FyH[h1Pos] += FyOnH1
        FxH[h2Pos] += FxOnH2
        FyH[h2Pos] += FyOnH2

    return FxO, FyO, FxH, FyH

# ================================================================================
# Simulation parameters
# ================================================================================
# General Parameters
N = 36
dt = 0.02 # timestep
# kWall = 10**-9 # stiffness of walls (N/Angstrom)
# kR = 6.5 * 1.6022e-19 * 10**10 # OH bond stiffness (N/Angstrom)
kWall = 50
kR = 6.5
wallProximity = 0.5 # how close atoms can be to the wall before being bounced back

# Atomic Properties
oMass = 16 # amu
hMass = 1.01 # amu
oRad = 1.52 # Angstroms
hRad = 1.2 # Angstroms
bondAngle = 104.45 * (np.pi / 180) # rad
bondLength = 0.9584 # Angstroms

# Parameters for Drawing
oColor = vp.color.red
hColor = vp.color.white
padding = 2 * oRad

# ================================================================================
# Initialization
# ================================================================================

# arrays for position: there are two different sets of arrays, one for oxygens
# and one for hydrogens. o[i]'s Hydrogen's are at h[2*i] and h[2*i + 1]. oDirs
# is the angle the oxygen molecule is rotated in radians from the horizontal (i.e right is 0)
oX = np.zeros(N, float)
oY = np.zeros(N, float)
oDir = np.zeros(N, float)
hX = np.zeros(2 * N, float)
hY = np.zeros(2 * N, float)

# arrays for velocity
vxO = np.array([0.5 - random() for i in range(N)], float)
vyO = np.array([0.5 - random() for i in range(N)], float)
vxmidO = np.zeros(N, float)
vymidO = np.zeros(N, float)
# instantiating the hydrogens with the same speed as their water
vxH = np.array([vxO[int(i / 2)] for i in range(2 * N)], float)
vyH = np.array([vyO[int(i / 2)] for i in range(2 * N)], float)
vxmidH = np.zeros(2 * N, float)
vymidH = np.zeros(2 * N, float)

t = 0

# lists of atoms (for redrawing)
os = []
hs = []

# parameters for spacing molecules
numRows = int(np.ceil(np.sqrt(N)))
moleculesPerRow = int(np.ceil(N / numRows))
dx = 1 if moleculesPerRow <= 1 else (w - 2*padding)/(moleculesPerRow - 1)
dy = 1 if numRows <= 1 else (w - 2*padding)/(numRows - 1)

# initializing molecules
for i in range(numRows):
    for j in range(min(moleculesPerRow, N - moleculesPerRow * i)):
        # initializing oxygen atom
        oLocation = i * moleculesPerRow + j

        xCoord = j * dx + padding
        yCoord = i * dy + padding
        mcAngle = random() * 2 * np.pi

        oX[oLocation] = xCoord
        oY[oLocation] = yCoord
        oDir[oLocation] = mcAngle

        # initializing accompanying hydrogens
        h1Location = 2 * oLocation
        h2Location = h1Location + 1

        h1X, h1Y, h2X, h2Y = calculateHPositions(xCoord, yCoord, mcAngle)
        hX[h1Location] = h1X
        hY[h1Location] = h1Y
        hX[h2Location] = h2X
        hY[h2Location] = h2Y

        # drawing in molecule
        os.append(vp.sphere(pos=vp.vector(xCoord, yCoord, 0), radius=oRad, color=oColor))
        hs.append(vp.sphere(pos=vp.vector(h1X, h1Y, 0), radius=hRad, color=hColor))
        hs.append(vp.sphere(pos=vp.vector(h2X, h2Y, 0), radius=hRad, color=hColor))

# Perform a half step
FxO, FyO, FxH, FyH = force(oX, oY, oDir, hX, hY)
vxmidO = vxO + 0.5 * dt * FxO / oMass
vymidO = vyO + 0.5 * dt * FyO / oMass
vxmidH = vxH + 0.5 * dt * FxH / hMass
vymidH = vyH + 0.5 * dt * FyH / hMass

# ================================================================================
# Main simulation loop
# ================================================================================

while True:

    vp.rate(5 / dt)

    # Verlet algorithm
    oX += vxmidO * dt
    oY += vymidO * dt
    hX += vxmidH * dt
    hY += vymidH * dt
    FxO, FyO, FxH, FyH = force(oX, oY, oDir, hX, hY)
    # vx = vxmid + 0.5 * Fx / m * dt
    # vy = vymid + 0.5 * Fy / m * dt
    vxmidO += FxO / oMass * dt
    vymidO += FyO / oMass * dt
    vxmidH += FxH / hMass * dt
    vymidH += FyH / hMass * dt

    # update atom positions
    for i in range(N):
        os[i].pos = vp.vector(oX[i], oY[i], 0)
        hs[2 * i].pos = vp.vector(hX[2 * i], hY[2 * i], 0)
        hs[2 * i + 1].pos = vp.vector(hX[2 * i + 1], hY[2 * i + 1], 0)

    t += dt