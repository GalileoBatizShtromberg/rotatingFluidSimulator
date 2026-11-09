import pygame
import sys

from src.config import window_width, window_height, fps
from src.grid import PatternGenerator
from src.graphics import Renderer


def main():


    pygame.init()
    screen = pygame.display.set_mode((window_width,window_height))
    pygame.display.set_caption("not sure yet lets see")
    clock = pygame.time.Clock()

    renderer = Renderer()
    generator = PatternGenerator(window_width, window_height)
    testGrid = generator.interference_pattern()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frameSurface = renderer.render_field(testGrid)
        screen.blit(frameSurface, (0, 0))
        pygame.display.flip() #No clue what this means(swap buffers?)
        clock.tick(60) #Framerate

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

