
# THE SETUP
#   the water is modeled as two stacked layers: a warmer layer on top q1
#   and a colder layer on the bottom q2 . Each layer is
#   allowed to flow horizontally. Quasi-geostrophic is a simplified set
#   of equations for fluids that rotate fast enough(quasi meaning its not full 3d)
#
# THE STREAMFUNCTION  psi
#   instead of storing the velocity (u, v) at every cell, we store one number
#   psi per cell, and get the velocity from its derivatives
#         u = -d(psi)/dy        (x-velocity)
#         v = +d(psi)/dx        (y-velocity)
#   
#   Bonus is that any flow made this way is incompressible because derivatives are negative/pos

#   in a rotating fluid, psi is proportional to pressure (psi = p / (rho f0)) pressure over densityxcoriolis
#   which is why the flow goes around highs and lows instead of into them(along the map contours)
#
# VORTICITY  zeta = lap(psi)
#   how much the water is spinning at each point the Laplacian of psi(to me, a mathematical black box)
#
# POTENTIAL VORTICITY  q
#   q1 = zeta1 - F * theta,     q2 = zeta2 + F * theta,     theta = psi1 - psi2
#   GOATED SETUP BTW SHOUTOUT TO CHATGPT this math was not my derivation but it comes from the 
#   quasi-geostrophic equations 

#   pv is carried by the flow so each parcel of water keeps its q value as it moves (apart from
#   temp and friction). so the simulation only has to track q1 and q2
#   psi, velocity, temperature so (vorticity, speed) can be recomputed from them
#
# THETA 
#   theta = psi1 - psi2. Physically how high the boundary between the cold
#   bottom layer and the warm top layer sits. More warm water means lower
#   boundary means bigger theta. So BIG THETA = warm

#   Thermal-wind balance means if theta changes from the center to the rim, then
#   the top and bottom layers must move at different speeds and that speed
#   difference called shear is what eventually goes unstable and makes eddies.
#
# Each time step:
#   1. given q1, q2, solve for psi1, psi2, theta.     - invert()
#   2. use psi to move q along with the flow, and add heating
#      and friction which gives dq/dt.                           - rhs()
#   3. update q using dq/dt over a small time DT                - step()
#
#
#   dq/dt = -J(psi, q)      q is carried by the flow (advection) - arakawa_jacobian()
#           relax       heating/cooling which pulls theta back toward theta_eq
#           r_ek zeta2    floor friction, bottom layer only  ekman drag
#           nu lap(zeta)  viscosity 
# =============================================================================

import numpy as np
# scipy.sparse is tools for sparse matrices which are huge matrices that are almost
# all zero and scipy magically stores only the non-zero entries, saving lots of memory.
import scipy.sparse as sp
# splu = sparse LU decomposition which is used to solve the big linear systems in invert
from scipy.sparse.linalg import splu
from scipy.ndimage import map_coordinates

from src.config import (
    NX, NY, H, DT, F_COUPLING, DTHETA, TAU_RELAX, R_EKMAN, NU,
    R_INNER, R_TANK, TAU_DYE, SEED_AMP,
)

#OPERATORS


# THE LAPLACIAN
# In 1D the second derivative on a grid with spacing H is approximated by
#     f''(x) ~ ( f(x+H) - 2 f(x) + f(x-H) ) / H^2
# obviously taylor-expand f(x+H) and f(x-H), add them, and the first-derivative terms
# cancel (thank you claude)
#
# The 2D Laplacian is d^2/dx^2 + d^2/dy^2, so add the x and y versions:
#     lap(a)[i,j] ~ ( a[i+1,j] + a[i-1,j] + a[i,j+1] + a[i,j-1] - 4 a[i,j] ) / H^2
# This is the 5-points for the laplacian since oyu need neighbors
#
# The obvious way to code that is a double for-loop over i and j. instead we
# use shifted slices so numpy does all cells at once because claude told me to
#
#     a[1:-1, 1:-1]   = every interior cell              -> a[i,   j  ]
#     a[2:,   1:-1]   = same cells shifted by +1 in i    -> a[i+1, j  ]
#     a[:-2,  1:-1]   = shifted by -1 in i               -> a[i-1, j  ]
#     a[1:-1, 2:  ]   = shifted by +1 in j               -> a[i,   j+1]
#     a[1:-1, :-2 ]   = shifted by -1 in j               -> a[i,   j-1]
#
# all 5 slices have the same shape (126, 126) and the element at position
# [k, l] in each one is the corresponding neighbour of the same interior cell since it was shifted
# so computing the slices element-by-element computes the stencil for all interior cells in one line.
#
# an example would be a = [10, 20, 40, 70]
#     a[2:]   = [40, 70]     (right neighbours of middle cells)
#     a[:-2]  = [10, 20]     (left neighbours of middle cells)
#     a[1:-1] = [20, 40]     (middle cells)
#     a[2:] + a[:-2] - 2*a[1:-1] = [10, 10]   becomes second difference at middle cells
#
# The outermost ring of cells is skipped for obvious reasons

def lap(a: np.ndarray) -> np.ndarray:
    # np.zeros_like(a) = an array of zeros with the same shape and type as a.
    out = np.zeros_like(a)
    # Fill only the interior using the shifted-slice trick described above.
    out[1:-1, 1:-1] = (
        a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2] - 4.0 * a[1:-1, 1:-1]
    ) / H**2
    return out


# 
# THE JACOBIAN  J(psi, q) = psi_x * q_y - psi_y * q_x
#  psi_x means d(psi)/dx, etc
#
# meaing substitute u = -psi_y and v = psi_x:
#     J(psi, q) = u * q_x + v * q_y   =   (velocity) dot product (gradient of q)

# This is how fast q changes at a fixed spot because the
# flow is carrying in water with a different q. Its advection
#     dq/dt = -J(psi, q)
#
#
# according to deareset claude 
# there are 3 ways to write J that are identical in calculus but give different answers on grid:
#     J++ = psi_x q_y - psi_y q_x                   (the direct formula)
#     J+x = d/dx(psi q_y) - d/dy(psi q_x)            (product-rule rearrangement)
#     Jx+ = d/dy(q psi_x) - d/dx(q psi_y)            (another rearrangement)

# Each one alone slowly creates or destroys energy because of grid errors like the boundaries
# which can make the simulation blow up or go artificially smooth.
# arakawa(hence arakawa) proved that the average of all three exactly conserves energy and
# enstrophy(total vorticity^2) on the grid



def arakawa_jacobian(psi: np.ndarray, q: np.ndarray) -> np.ndarray:
   
    # psi[e] is the same as psi[2:, 1:-1] where e is a tuple

    #     c  = center      (i,   j  )
    #     e  = east        (i+1, j  )      w  = west        (i-1, j  )
    #     n  = north       (i,   j+1)      s  = south       (i,   j-1)
    #     ne = north-east  (i+1, j+1)      nw = north-west  (i-1, j+1)
    #     se = south-east  (i+1, j-1)      sw = south-west  (i-1, j-1)
    #       
    # should note that tuple(None, 1) is slice[ :1]

    c  = (slice(1, -1), slice(1, -1))
    e  = (slice(2, None), slice(1, -1));  w  = (slice(None, -2), slice(1, -1))
    n  = (slice(1, -1), slice(2, None));  s  = (slice(1, -1), slice(None, -2))
    ne = (slice(2, None), slice(2, None));  nw = (slice(None, -2), slice(2, None))
    se = (slice(2, None), slice(None, -2)); sw = (slice(None, -2), slice(None, -2))

    # normal formula
    jpp = ((psi[e] - psi[w]) * (q[n] - q[s]) - (psi[n] - psi[s]) * (q[e] - q[w]))
    # J+x the formula using like theneighbors with the product rule rearangement
    jpx = (psi[e] * (q[ne] - q[se]) - psi[w] * (q[nw] - q[sw])
           - psi[n] * (q[ne] - q[nw]) + psi[s] * (q[se] - q[sw]))
    # Jx+ : the other diagonal neighbors
    jxp = (psi[ne] * (q[n] - q[e]) - psi[sw] * (q[w] - q[s])
           - psi[nw] * (q[n] - q[w]) + psi[se] * (q[e] - q[s]))

    
    out = np.zeros_like(psi)
    # average the three (divide by 3) and apply the 1/(4H^2) factor that would appear on each individual otherwise
    out[c] = (jpp + jpx + jxp) / (12.0 * H**2)
    return out


# ------------------------------------------------------------------- model
class RotatingTwoLayer:
    def __init__(self, r_grid: np.ndarray):
        # r_grid[i, j] = distance of cell (i, j) from the center, in cells.
        # self.fluid is the same true or false for iswater 

        self.fluid = (r_grid > R_INNER) & (r_grid < R_TANK)
        # f is just is fluid
        f = self.fluid

       
        # Solving lap(T) = 0 in polar coordinates symettrically gives
        # T = A + B * ln(r) and then made so
        # s = 0 at the can and s = 1 at the rim note the ln so it gets warmer logarithmically
        #

        # np.clip forces r into rinner rtank first, so cells inside the can
        # get s = 0, cells outside the rim get s = 1, and we never take ln(0)
        rr = np.clip(r_grid, R_INNER, R_TANK)
        # np.log is the natural log (ln), applied to every element.
        self.s = np.log(rr / R_INNER) / np.log(R_TANK / R_INNER)   # 0 at can, 1 at rim

      
        # At the can: s = 0 so theta_bg = +DTHETA/2  (cold).
        # At the rim: s = 1 -> theta_bg = -DTHETA/2  (warm).
        # "bg" = background: the smooth, perfectly round starting profile.
        self.theta_bg = DTHETA * (self.s - 0.5)

        # theta_eq = the "equilibrium" profile the heating/cooling pulls toward.
        # here it's the same array as theta_bg 
        self.theta_eq = self.theta_bg
        # (lap - 2F) theta background
        # precomputed because invert() needs it every single step.
        self.Lbg = lap(self.theta_bg) - 2.0 * F_COUPLING * self.theta_bg




        # sparse elliptic operators(like lap) on fluid cells only dropping the coupling to
        # non-fluid cells are Dirichlet boundary something (constant-psi) walls on the can and rim

        # lap() above takes psi and gives q. But in invert() we need the
        # inverse given q, find the psi whose Laplacian is q
        # that's solving a differential equation the Poisson 
        # on the grid, it becomes a system of linear equations, one equation per cell:
        #     (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1] - 4 psi[i,j]) / H^2 = q[i,j]

        # that maeans its a matrix L  at psi = q  where psi and q are
        # flattened into 1D vectors with all cells in a row
        #
        # L is a 12,000 x 12,000 matrix, but each row has at most 5 non-zero
        # entries (a cell and its 4 neighbours)
        # use a SPARSE matrix that stores only the non-zeros.
        # f.ravel() flattens the 2D (128, 128) and does that by making it into one 1D list of 16,384
        # values: 
        
        # cell [i, j] goes to position i*NY + j.

        # np.flatnonzero(...) returns the positions where that list is true.
        # So idx = the flattened positions of all the water cells.

        idx = np.flatnonzero(f.ravel())
        # the 1d second-difference matrix is -2 on the main diagonal, 1 on the
        # diagonals just above and below. 

        # Multiplying it by a vector [f0, f1, f2, f3] gives (f[k-1] - 2 f[k] + f[k+1])
        # at each k or the 1D stencil from above 

        # sp.diags(values, offsets, shape): put values[0] on diagonal offsets[0],
        # etc. Offset 0 = main diagonal, -1 = just below, +1 = just above.
        Dx = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(NX, NX))
        Dy = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(NY, NY))

        # this builds the 2d laplacian matrix (the mystery multipliers for the whole grid)
        # sp.kron is the gives the kroneker product which is 
        #   kron(Dx, I) applies Dx along the i direction, at every j separately
        #               -> the d^2/dx^2 part
        #   kron(I, Dy) applies Dy along the j direction, at every i separately
        #               -> the d^2/dy^2 part

        # this works because the flattening order is i*NY + j meaning consecutive
        # entries differ in j, and entries NY apart differ in i 
        
        # Adding them and dividing by H^2 gives the 5-point Laplacian as a matrix:
        # L_full at a.ravel() equals lap(a).ravel() at interior cells.
        L_full = (sp.kron(Dx, sp.identity(NY)) + sp.kron(sp.identity(NX), Dy)) / H**2



        #   .tocsr() is To Compressed Sparse Row format, a storage layout
        #            that's fast at picking out rows.
        #   [idx]    keeps only the water-cell rows (one equation per water cell)
        #   [:, idx] keeps only the water-cell columns (one unknown per water cell)
        # Throwing away the columns for non-water cells is the same as treating
        # psi = 0 in every non-water cell
        
        L = L_full.tocsr()[idx][:, idx]
        # LU DECOMPOSITION: splitting a matrix into L(ower triangular) times
        # U(pper triangular). The expensive factoring is done ONCE here, because the matrix never changes; after
        # that, each .solve(rhs) call in invert() is faster
        
        # lu_bt: solves  lap(psi_bt) = rhs                (Poisson equation)
        self.lu_bt = splu(L.tocsc())           # lap(psi_bt) = q_bt

        
        # lu_bc: solves  (lap - 2F) theta' = rhs          (Helmholtz equation)
        # "L - 2F * identity" = the Laplacian matrix minus 2F on the diagonal.
        # len(idx) = number of water cells.
        self.lu_bc = splu((L - 2.0 * F_COUPLING * sp.identity(len(idx))).tocsc())  # (lap - 2F) theta' = rhs

        # semi-Lagrangian grid for the dye
        # The (i, j) index of every cell, as two 2D arrays (see meshgrid in main.py).
        self.ii, self.jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")

        # kind of a black box tbh- I get the most vague understanding of how the code follows the math


    def initial_state(self):
        # starting state to be theta = theta_bg with no barotropic flow
        #   q1 - q2 = (lap - 2F) theta   -> set it to (lap - 2F) theta_bg = Lbg
        #   q1 + q2 = lap(psi1 + psi2)   -> set it to 0 (no barotropic flow)
        # Solving those two: q1 = +Lbg/2, q2 = -Lbg/2.
        

        q_bc = self.Lbg * self.fluid                # q1 - q2 of the mean state
        q1 = 0.5 * q_bc
        q2 = -0.5 * q_bc


        # the largest absolute value of q_bc
        scale = np.abs(q_bc).max()
        # a random number generator 
        rng = np.random.default_rng()
        # rng.standard_normal(shape) = normal distribution 
        q1 = q1 + SEED_AMP * scale * rng.standard_normal(q1.shape) * self.fluid
        q2 = q2 + SEED_AMP * scale * rng.standard_normal(q2.shape) * self.fluid
        
        # initialize dye
        dye = self.s.copy()
        return q1, q2, dye

    def invert(self, q1, q2):
        # Given q1 and q2, find psi1 and psi
        #     q1 = lap(psi1) - F (psi1 - psi2)
        #     q2 = lap(psi2) + F (psi1 - psi2)
        # are tangled together, my boy claude separated them thus:

        #   ADD:      q1 + q2 = lap(psi1 + psi2)
        #             Define psi_bt = (psi1 + psi2) / 2  barotropic
        #                 lap(psi_bt) = (q1 + q2) / 2           <- Poisson
        #
        #   SUBTRACT: q1 - q2 = lap(psi1 - psi2) - 2F (psi1 - psi2)
        #             theta = psi1 - psi2 baroclinic
        #                 (lap - 2F) theta = q1 - q2            <- Helmholtz
        #
        # rebuild as
        #     psi1 = psi_bt + theta/2,    psi2 = psi_bt - theta/2
        

        f = self.fluid
        psi_bt = np.zeros((NX, NY))
        # (0.5 * (q1 + q2))[f] picks the water-cell values as a 1D array, returns the answer
        # as a 1D array in the same order, and psi_bt[f] writes each
        # value back into its cell.

        psi_bt[f] = self.lu_bt.solve(0.5 * (q1 + q2)[f])

        # theta is split as theta = theta_bg + theta_prime where theta_prime is the
        # departure from the smooth background and is 0 on the walls. That way
        # the walls can be forced easily

        # plugging into (lap - 2F) theta = q1 - q2 gives
        #     (lap - 2F) theta' = (q1 - q2) - (lap - 2F) theta_bg = (q1 - q2) - Lbg
        # which is the right-hand side passed to solve() below
        theta = self.theta_bg.copy()                       # carries the wall values
        theta[f] += self.lu_bc.solve((q1 - q2)[f] - self.Lbg[f])

        # return psi1, psi2, theta
        return psi_bt + 0.5 * theta, psi_bt - 0.5 * theta, theta

    # right-hand side of  dq/dt = (stuff) this function computes how
    # fast q1 and q2 are changing right now, given their current values.
    def rhs(self, q1, q2):
        f = self.fluid
        
        #get the streamfunctions from the PV.
        psi1, psi2, theta = self.invert(q1, q2)

        # get vorticity of each layer.
        zeta1, zeta2 = lap(psi1), lap(psi2)

        # F * theta cuz stretching 
        stretch = F_COUPLING * theta

        # rebuild q1 = zeta1 - F theta, etc.
        qf1 = zeta1 - stretch          # PV on the *whole* array (walls included)
        qf2 = zeta2 + stretch

        # relaxation wherever theta differs from
        # its equilibrium theta_eq, nudge it back, closing a fraction 1/TAU_RELAX
        # of the gap per dt.
        #     d(theta)/dt ~ -(theta - theta_eq) / TAU_RELAX.
        relax = F_COUPLING * (theta - self.theta_eq) / TAU_RELAX

        # top layer:     advection          + heating  + viscosity
        dq1 = -arakawa_jacobian(psi1, qf1) + relax + NU * lap(zeta1)
        # Bottom layer:  advection          - heating  - floor friction    + viscosity
        dq2 = -arakawa_jacobian(psi2, qf2)  - relax     - R_EKMAN * zeta2 + NU * lap(zeta2)


        #the outside is set to fluid,  psi1 and theta are returned too so the
        return dq1 * f, dq2 * f, psi1, theta

    # RUNGE-KUTTA 4 (RK4)
    def step(self, q1, q2):
        k1a, k1b, psi1, theta = self.rhs(q1, q2)
        k2a, k2b, _, _ = self.rhs(q1 + 0.5 * DT * k1a, q2 + 0.5 * DT * k1b)
        k3a, k3b, _, _ = self.rhs(q1 + 0.5 * DT * k2a, q2 + 0.5 * DT * k2b)
        k4a, k4b, _, _ = self.rhs(q1 + DT * k3a, q2 + DT * k3b)
        q1 = q1 + DT / 6.0 * (k1a + 2 * k2a + 2 * k3a + k4a)
        q2 = q2 + DT / 6.0 * (k1b + 2 * k2b + 2 * k3b + k4b)

        # note that psi1 and theta come from the k1 call so they describe the
        # start of this step, not after the update.
        return q1, q2, psi1, theta

    def advect_dye(self, dye, psi1):

        # Semi-Lagrangian method has flaws but its fine for tracer
        
    
        # velocity from the streamfunction, with central differences:
        #     u = -d(psi)/dy ~ -(psi[i, j+1] - psi[i, j-1]) / (2H)
        #     v = +d(psi)/dx ~ +(psi[i+1, j] - psi[i-1, j]) / (2H)
        # Same shifted-slice trick as lap() with edges at zero

        u = np.zeros_like(psi1)
        v = np.zeros_like(psi1)
        u[1:-1, 1:-1] = -(psi1[1:-1, 2:] - psi1[1:-1, :-2]) / (2.0 * H)
        v[1:-1, 1:-1] =  (psi1[2:, 1:-1] - psi1[:-2, 1:-1]) / (2.0 * H)

        # the semi lagrancian
        di = self.ii - u * DT / H
        dj = self.jj - v * DT / H

        # find the dye at those departure points using cubic interpolation
        dye = map_coordinates(dye, [di, dj], order=3, mode="nearest")

        # pull dye to starting point(means mass is not kept constant tho)
        dye += DT * (self.s - dye) / TAU_DYE               
        # cubic interpolation can slightly overshoot
        # the dye back into [0, 1].
        dye = np.clip(dye, 0.0, 1.0)
        return dye

    def temperature_for_display(self, theta):
        return np.clip(0.5 + theta / DTHETA, 0.0, 1.0)
