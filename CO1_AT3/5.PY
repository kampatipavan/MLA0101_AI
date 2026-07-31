import heapq

# Goal State
goal = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]


# Heuristic Function (Misplaced Tiles)
def heuristic(state):
    count = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != goal[i][j]:
                count += 1
    return count


# Find Blank Tile
def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j


# Generate Next States
def neighbors(state):

    x, y = find_blank(state)

    moves = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    result = []

    for dx, dy in moves:

        nx = x + dx
        ny = y + dy

        if 0 <= nx < 3 and 0 <= ny < 3:

            new_state = [row[:] for row in state]

            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]

            result.append(new_state)

    return result


# Convert Board to Tuple
def board_tuple(state):
    return tuple(tuple(row) for row in state)


# A* Algorithm
def astar(start):

    pq = []

    heapq.heappush(pq, (heuristic(start), 0, start, []))

    visited = set()

    while pq:

        f, g, state, path = heapq.heappop(pq)

        if state == goal:
            return path + [state]

        key = board_tuple(state)

        if key in visited:
            continue

        visited.add(key)

        for next_state in neighbors(state):

            if board_tuple(next_state) not in visited:

                heapq.heappush(
                    pq,
                    (
                        g + 1 + heuristic(next_state),
                        g + 1,
                        next_state,
                        path + [state]
                    )
                )

    return None


# Initial State
start = [
    [1, 2, 3],
    [4, 0, 6],
    [7, 5, 8]
]


solution = astar(start)

if solution:

    print("Solution Found!\n")

    step = 0

    for state in solution:

        print("Step", step)

        for row in state:
            print(row)

        print()

        step += 1

    print("Total Moves =", step - 1)

else:
    print("No Solution Exists")