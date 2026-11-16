import pygame
import sys
import numpy as np
# map_coordinates is the interpolation function
from scipy.ndimage import map_coordinates


from src.config import window_width, window_height, fps, NX, NY, STEPS_PER_FRAME, DT, OMEGA_TANK
from src.grid import PatternGenerator
from src.graphics import Renderer
from src.solver import RotatingTwoLayer


def main():
    pygame.init()
    # returns a "Surface" object thats pixels
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("SPINlab Rotating Convection Tank  [SPACE: dye / temperature,  R: reset]")
    clock = pygame.time.Clock()

    # renderer in the graphics section knows array in range [0, 1] into a coloured picture
    renderer = Renderer()

    
    generator = PatternGenerator(NX, NY)


    # r_grid[i, j] = distance (in grid cells) from cell (i, j) to the tank center.
    _, _, r_grid, _, _, _ = generator.create_tank_geometry()

    # build the physics model 
    model = RotatingTwoLayer(r_grid)

    # build the initial state of the simulation:
    #   q1 and q2 are the map of the potential vorticity of the top and bottom water layers.
    #            these are the main array variables that solver evolves 
    q1, q2, dye = model.initial_state()
    # theta =  a copy of the initial temperature field for display to evolve
    theta = model.theta_bg.copy()
    show_dye = True
    # angle of rotation
    angle = 0.0

    # grid coordinates calculated once
   
    i, j = np.meshgrid(
        np.arange(NX),
        np.arange(NY),
        indexing="ij"
    )

    # set xc and yc to x center and y center with NX NY equal 0..127  center is at
    # 63.5 halfway between cell 63 and cell 64 which works cuz map coords
    xc = (NX - 1) / 2
    yc = (NY - 1) / 2

    # this sets x[a, b], y[a, b] = position of cell (a, b) measured from the center so that coords are normal
    # mind you this is why meshgrid matters because now its 

    # [-1, 0 , 1]           [1, 1 , 1]
    # [-1, 0 , 1]           [0, 0 , 0]
    # [-1, 0 , 1]           [-1,-1,-1]
    
    x = i - xc
    y = j - yc

    # main loop runs 60 frames per second 
    running = True
    while running:
        # pygame collects everything that happened since last frame into a list of events
        # which we go through
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # space switches between dye and temp 
                if event.key == pygame.K_SPACE:
                    show_dye = not show_dye
                # r is reset 
                elif event.key == pygame.K_r:
                    angle = 0.0
                    q1, q2, dye = model.initial_state()

        # we do multiple steps(which is dt+1) per frame so the simulation runs faster than
        # one step per frame while keeping steps small because of that law 
        for _ in range(STEPS_PER_FRAME):
            # step() returns the new q1, q2, plus psi1 (the top layer's streamfunction with
            # its velocity for modeling) and theta (which is the top difference bottom and 
            # is essential temp because large difference means top pushing down(cold) and 
            # small difference is top not pushing down(warm))
            q1, q2, psi1, theta = model.step(q1, q2)
            # moves the dye
            dye = model.advect_dye(dye, psi1)

        # angle is used for camera here we just rotate it one step in a normal way
        angle += OMEGA_TANK * DT * STEPS_PER_FRAME
        # keep angle between 0 and 2pi cuz its periodic
        angle %= 2 * np.pi
        # rotation transformation helper 
        c = np.cos(angle)
        s = np.sin(angle)

        # coordinates in the rotating tank frame using some matrix 
        #     [ c  -s ] [ x ]       c and s calculated above
        #     [ s   c ] [ y ]
        #
        # then adding the center xc and yc back to turn the change into an array
        # index off of x and y which are also arrays so it changes the whole thing at once
        tank_i = xc + c*x - s*y
        tank_j = yc + s*x + c*y

        #keeps track of what to show        
        field = dye if show_dye else model.temperature_for_display(theta)

        # transform data into the regular camera fram
        # map_coordinates builds an array with normal indexing from the sin wierd numbers
        field_lab = map_coordinates(
            field,
            [tank_i, tank_j],
            order=1,
            mode="nearest"
        )

        #  model.fluid is the array marking water vs container
        #
        # screen.blit copies that image onto the window canvas
        # with its top-left corner at given pixel 
        screen.blit(
            renderer.render_field(field_lab, model.fluid),
            (0, 0)
        )

        # pygame draws onto a hidden buffer first so you never see a half-drawn
        # frame and flip puts it on the screen
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
