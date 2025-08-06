# MazMagic_SplitView_Final.py
# Enhanced Exit Marker & Balanced Minimap Tones

import pygame
import moderngl
import numpy as np
from pygame.locals import *
from pyrr import Matrix44, Vector3
from MazGenerator import generate_maze

WIDTH, HEIGHT = 1000, 600
TILE_SIZE = 2.0
MAZE_ROWS, MAZE_COLS = 21, 21

pygame.init()
pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
ctx = moderngl.create_context()
ctx.enable(moderngl.DEPTH_TEST)

# Shaders
vertex_shader = '''
#version 330
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
in vec3 in_vert;
void main() {
    gl_Position = projection * view * model * vec4(in_vert, 1.0);
}'''

fragment_shader = '''
#version 330
out vec4 fragColor;
void main() {
    fragColor = vec4(0.2, 0.6, 1.0, 1.0);
}'''

prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

cube_vertices = np.array([
    -1, -1, -1,
     1, -1, -1,
     1,  1, -1,
    -1,  1, -1,
    -1, -1,  1,
     1, -1,  1,
     1,  1,  1,
    -1,  1,  1,
], dtype='f4')

cube_indices = np.array([
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4,
    0, 1, 5, 5, 4, 0,
    2, 3, 7, 7, 6, 2,
    0, 3, 7, 7, 4, 0,
    1, 2, 6, 6, 5, 1,
], dtype='i4')

vbo = ctx.buffer(cube_vertices.tobytes())
ibo = ctx.buffer(cube_indices.tobytes())
vao = ctx.vertex_array(prog, [(vbo, '3f', 'in_vert')], index_buffer=ibo)

maze = generate_maze(MAZE_ROWS, MAZE_COLS)
exit_pos = (MAZE_ROWS - 2, MAZE_COLS - 2)

camera_pos = Vector3([TILE_SIZE + 1.0, 1.0, TILE_SIZE + 1.0])
camera_front = Vector3([0.0, 0.0, -1.0])
camera_up = Vector3([0.0, 1.0, 0.0])
yaw, pitch = -90.0, 0.0
speed = 0.1
sensitivity = 2.0

projection = Matrix44.perspective_projection(70.0, (WIDTH * 0.7) / HEIGHT, 0.1, 100.0)
prog['projection'].write(projection.astype('f4').tobytes())

# Helper
def is_walkable(pos):
    col = int(pos.x // TILE_SIZE)
    row = int(pos.z // TILE_SIZE)
    if 0 <= row < MAZE_ROWS and 0 <= col < MAZE_COLS:
        return maze[row][col] == 0
    return False

# Main Loop
clock = pygame.time.Clock()
running = True

while running:
    dt = clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    front = Vector3([
        np.cos(np.radians(yaw)) * np.cos(np.radians(pitch)),
        np.sin(np.radians(pitch)),
        np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
    ])
    camera_front = front.normalized

    move_vector = Vector3([0.0, 0.0, 0.0])

    if keys[pygame.K_w]:
        move_vector += camera_front * speed
    if keys[pygame.K_s]:
        move_vector -= camera_front * speed

    right = camera_front.cross(camera_up).normalized
    if keys[pygame.K_a]:
        move_vector -= right * speed
    if keys[pygame.K_d]:
        move_vector += right * speed

    next_pos = camera_pos + move_vector
    if is_walkable(next_pos):
        camera_pos = next_pos

    if keys[pygame.K_LEFT]:
        yaw -= sensitivity
    if keys[pygame.K_RIGHT]:
        yaw += sensitivity

    # 3D Viewport
    ctx.viewport = (0, 0, int(WIDTH * 0.7), HEIGHT)
    ctx.clear(0.0, 0.0, 0.0)
    view = Matrix44.look_at(camera_pos, camera_pos + camera_front, camera_up)
    prog['view'].write(view.astype('f4').tobytes())

    for row in range(MAZE_ROWS):
        for col in range(MAZE_COLS):
            if maze[row][col] == 1:
                model = Matrix44.from_translation(Vector3([
                    col * TILE_SIZE,
                    1.0,
                    row * TILE_SIZE
                ])) * Matrix44.from_scale(Vector3([TILE_SIZE / 2, 1.0, TILE_SIZE / 2]))
                prog['model'].write(model.astype('f4').tobytes())
                vao.render()

    # 2D Minimap Viewport
    ctx.viewport = (int(WIDTH * 0.7), 0, int(WIDTH * 0.3), HEIGHT)
    ctx.screen.use()
    ctx.disable(moderngl.DEPTH_TEST)

    grid_shader = ctx.program(
        vertex_shader='''
            #version 330
            in vec2 in_vert;
            in vec4 in_color;
            out vec4 v_color;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                v_color = in_color;
            }''',
        fragment_shader='''
            #version 330
            in vec4 v_color;
            out vec4 fragColor;
            void main() {
                fragColor = vec4(v_color.rgb * 0.5, 1.0);
            }'''
    )

    verts = []
    colors = []
    for row in range(MAZE_ROWS):
        for col in range(MAZE_COLS):
            x = -1.0 + (2.0 * col / MAZE_COLS)
            y = -1.0 + (2.0 * row / MAZE_ROWS)
            size_x = 2.0 / MAZE_COLS
            size_y = 2.0 / MAZE_ROWS

            verts.extend([
                x, y,
                x + size_x, y,
                x, y + size_y,
                x + size_x, y + size_y
            ])

            if (row, col) == exit_pos:
                color = [1.0, 0.0, 0.0, 1.0]
            elif maze[row][col] == 1:
                color = [0.0, 1.0, 0.0, 1.0]
            else:
                color = [0.0, 0.0, 0.0, 1.0]
            colors.extend(color * 4)

    # Player marker
    px = -1.0 + (2.0 * camera_pos.x / (MAZE_COLS * TILE_SIZE))
    py = -1.0 + (2.0 * camera_pos.z / (MAZE_ROWS * TILE_SIZE))
    size = 0.04
    verts.extend([
        px - size, py - size,
        px + size, py - size,
        px - size, py + size,
        px + size, py + size
    ])
    colors.extend([1.0, 1.0, 1.0, 1.0] * 4)

    vbo = ctx.buffer(np.array(verts, dtype='f4').tobytes())
    cbo = ctx.buffer(np.array(colors, dtype='f4').tobytes())

    vao2d = ctx.vertex_array(grid_shader, [(vbo, '2f', 'in_vert'), (cbo, '4f', 'in_color')])
    vao2d.render(moderngl.TRIANGLE_STRIP, vertices=len(verts)//2)

    ctx.enable(moderngl.DEPTH_TEST)
    pygame.display.flip()

pygame.quit()