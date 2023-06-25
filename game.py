import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from pygame.display import get_surface

# game type
game = 1

# number of mines
n = 30

pygame.init()
font = pygame.font.SysFont("monospace", 20)


class Direction(Enum):
    R = 1
    L = 2
    U = 3
    D = 4


Point = namedtuple('Point', 'x, y')

# colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLUE3 = (16, 78, 139)
BLACK = (0, 0, 0)
GREEN1 = (102, 205, 0)
GREEN2 = (69, 139, 0)

BLOCK_SIZE = 20
SPEED = 150


class SnakeAI:

    def __init__(self, w=640, h=480):
        self.mines = None
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        self.prev_distance = 10000
        self.current_distance = 0

    def reset(self):
        self.direction = Direction.U
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head]
        self.score = 0
        self.food = None
        self._place_food()

        if game == 3:
            self.mines = None
            self.place_mine(n)

        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        try:
            if self.food in self.mines:
                self._place_food()
        except:
            pass

    def update_food(self):

        while True:
            direction = random.choice(['u', 'd', 'l', 'r'])
            x, y = self.food
            if direction == 'u':
                y -= BLOCK_SIZE
            elif direction == 'd':
                y += BLOCK_SIZE
            elif direction == 'l':
                x -= BLOCK_SIZE
            elif direction == 'r':
                x += BLOCK_SIZE

            up = Point(x, y-20)
            down = Point(x, y+20)
            left = Point(x-20, y)
            right = Point(x+20, y)

            if self.is_collision(up) and self.is_collision(down) and self.is_collision(left) and self.is_collision(right):
                self.food = Point(x, y)
                break

            if Point(x, y) not in self.snake and not self.is_collision(Point(x, y)):
                self.food = Point(x, y)
                break

    def place_mine(self, n):
        self.mines = [Point(random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
               random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE) for i in range(0, n)]

        for pt in self.mines:
            if pt in self.mines == self.food:
                return self.place_mine(n)

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.frame_iteration > 30 * len(self.snake):
            reward = -1

        if self.head == self.food:
            self.score += 1
            reward = 15
            self._place_food()
        else:
            self.snake.pop()

        if game == 2:
            if self.frame_iteration % 5 == 0:
                self.update_food()

        self.current_distance = np.sqrt((self.head.x-self.food.x)**2+(self.head.y-self.food.y)**2)

        if self.prev_distance < self.current_distance:
            reward += 1

        else:
            reward += -1

        self.prev_distance = self.current_distance

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        if game == 3:
            if pt in self.mines[0:]:
                return True

        return False

    def _update_ui(self):
        self.display.fill(BLUE3)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        if game == 3:
            for pt in self.mines:
                pygame.draw.rect(self.display, BLACK, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):

        clock_wise = [Direction.R, Direction.D, Direction.L, Direction.U]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.R:
            x += BLOCK_SIZE
        elif self.direction == Direction.L:
            x -= BLOCK_SIZE
        elif self.direction == Direction.D:
            y += BLOCK_SIZE
        elif self.direction == Direction.U:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_surface_(self):
        snake = self.snake
        food = self.food
        surface = np.zeros((25, 33), dtype=int)
        surface[0,:] = -1
        surface[:, -1] = -1
        surface[-1,:] = -1
        surface[:, 0] = -1
        for pt in snake:
            surface[int(pt.y/20)][int(pt.x/20)] = 2

        surface[int(self.head.y/20)][int(self.head.x/20)] = 1

        surface[int(food.y/20)][int(food.x/20)] = 3

        if game == 3:
            mines = self.mines
            for pt in mines:
                surface[int(pt.y/20)][int(pt.x/20)] = -1

        return surface
