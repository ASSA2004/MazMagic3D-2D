import random

def generate_maze(rows, cols):
    # Ensure dimensions are odd
    if rows % 2 == 0:
        rows += 1
    if cols % 2 == 0:
        cols += 1

    # Initialize maze: 1 for walls, 0 for paths
    maze = [[1 for _ in range(cols)] for _ in range(rows)]

    # Directions: (dy, dx)
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    def is_valid(nr, nc):
        return 0 <= nr < rows and 0 <= nc < cols

    def carve_passages_from(r, c):
        maze[r][c] = 0
        random.shuffle(directions)

        for dy, dx in directions:
            nr, nc = r + dy, c + dx
            if is_valid(nr, nc) and maze[nr][nc] == 1:
                wall_y, wall_x = r + dy // 2, c + dx // 2
                maze[wall_y][wall_x] = 0
                maze[nr][nc] = 0
                carve_passages_from(nr, nc)

    # Start from cell (1,1)
    carve_passages_from(1, 1)

    # Make sure exit is clear
    maze[rows - 2][cols - 2] = 0
    return maze
