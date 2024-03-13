import pygame
from random import randint
import neat
import os

game_speed = 3

class Dino:
    pos = 50
    height = 50
    y_force = 0
    jump_force = 1

def eval_genomes(genomes, config):
    global dinos, ge, nets, points, game_speed
    pygame.init()

    
    #init vars
    screen_width = 700
    screen_height = 500
    ground = 50
    wall_pos = screen_width
    wall_height = 120
    wall_speed = 0.5
    gravity = 0.003
    ticks = 0
    #game_speed = 3 #Lower is faster (1 is fastest)
    points = 0

    #GENOMES
    dinos = []
    ge = []
    nets = []

    for genome_id, genome in genomes:
        dinos.append(Dino())
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    #dinos.append(Dino())


    window_size = (screen_width, screen_height)
    window = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Jumper")

    running = True

    def is_grounded(pos):
        if pos <= ground:
            pos = ground
            return True
        return False
    
    def get_distance(dino):
        return wall_pos-100

    while running:
        # Handle events/SPEED CHANGE
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and game_speed >= 2:
                    game_speed -= 1
                if event.key == pygame.K_LEFT:
                    game_speed += 1
            if event.type == pygame.QUIT:
                running = False

        if len(dinos) == 0:
            # If all dinos died, reset the game state
            eval_genomes(genomes, config)
            continue
        
        if len(dinos) == 1:
            print(points)

        # game loop (tick)
        ticks += 1
        if ticks % game_speed == 0:
            # Create a list to keep track of dinos that need to be removed
            to_remove = []
            
            # GAME LOGIC HERE
            wall_pos -= wall_speed
            
            for i, dino in enumerate(dinos):
                dino.pos += dino.y_force
                if not is_grounded(dino.pos):
                    dino.y_force -= gravity
                else:
                    dino.y_force = 0

                if wall_pos + screen_width <= 0:
                    wall_pos = randint(50, 800)
                    points += 1

                output = nets[i].activate((dino.pos, get_distance(dino)))

                if output[0] > 0.5 and is_grounded(dino.pos):
                    dino.y_force = dino.jump_force

                # Check for collision and mark dino for removal
                if dino.pos - 50 <= wall_height and 100 <= wall_pos + screen_width + 50 and 150 >= wall_pos + screen_width:
                    ge[i].fitness -= 1
                    to_remove.append(i)

            # Remove marked dinos from lists
            for index in sorted(to_remove, reverse=True):
                dinos.pop(index)
                ge.pop(index)
                nets.pop(index)

        # Draw objects and update display outside of the dino loop
        window.fill((0, 0, 0))
        for dino in dinos:
            pygame.draw.rect(window, (255, 0, 0), (100, (screen_height - dino.height) - dino.pos, 50, dino.height))
        pygame.draw.rect(window, (255, 255, 255), (wall_pos + screen_width, (screen_height - wall_height) - ground, 50, wall_height))
        pygame.draw.rect(window, (150, 150, 150), (0, screen_height - ground, screen_width, ground))
        pygame.display.flip()

    pygame.quit()

#NEAT:
def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.run(eval_genomes, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
