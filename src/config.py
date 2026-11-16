

window_width = 800
window_height = 800
fps = 60

#simulation grid size where each box holds the scalar values like temp 
NX, NY = 128, 128

# physical size of the box as if it were 1 meter everything elese is scaled to this as 1
LX, LY = 1.0, 1.0

# H is the width of one cell in our distance units
H = LX / NX

# radius tank is width/2  and minus 2 so that walls can exist
R_TANK  = (NY / 2.0) - 2.0   # outer wall radius
R_INNER = 6.0               # the radius of the middle thing


# Rotation enters through F = f0^2 / (g' * H_layer) = 1 / Ld^2,
# where Ld is the rossby deformation radius-  eddies apparently come out ~ 2 to 4 times Ld wide.

#  where f0 is coriolis parameter = 2 * rotation rate . faster spin is bigger f0.
#   g'      = gravity times the fractional density difference of layers
#    small density difference -> small g'
#  H_layer = thickness/height of a water layer

# all of those combine into one LD which is how strong is rotation
# compared stratification(diff between layers). spinning the tank faster would make Ld smaller

# LD = 0.045 units = 0.045 * 128 ~ 6 cells.
LD         = 0.045             

# F = 1 / 0.045^2 ~ 500. It's how strongly the two layers feel each other.
F_COUPLING = 1.0 / LD**2       # connection between the layers



# DTHETA is how big the height difference is between can and rim theta
DTHETA     = 0.025

# Relaxation time how long it takes to pull the temperature back toward its gradient profile.
TAU_RELAX  = 6.0             


# Ekman friction is subtractive drag from the tank floor frictioning on the bottom layer
# 0.15 per time unit means motions in the bottom layer lose most of their
# speed in 1/0.15 ~ 6.7 ;) dt units (like 330 steps) if nothing drives them.
R_EKMAN    = 0.15

# Viscosity
NU        = 1.0e-4            

# time stepping for RK4 + Arakawa Jacobian must stable for u*DT/H < ~1 cuz CFL 
DT         = 0.02

# how many physics steps to do per drawn frame 2 steps * 0.02 = 0.04 time
# units per frame so at 60 fps that's 2 and a bit time units per secon
STEPS_PER_FRAME = 2


TAU_DYE    = 60.0                 # slow relaxation back to the can/rim colours

# tiny random kick where baroclinic instability
# picks its own wavenumber out of this noise.
SEED_AMP   = 1.0e-3

# old names still imported by grid.py only old helpers use
t_hot, t_cold = 1.0, 0.0

OMEGA_TANK = 0.2   # Background angular velocity
