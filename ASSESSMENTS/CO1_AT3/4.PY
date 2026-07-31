import math
import random
import copy

ROWS = 6
COLS = 7

EMPTY = 0
PLAYER = 1
AI = 2


def create_board():
    return [[0 for _ in range(COLS)] for _ in range(ROWS)]


def print_board(board):
    print()
    for row in board:
        print(" ".join(str(x) for x in row))
    print("0 1 2 3 4 5 6\n")


def is_valid_location(board, col):
    return board[0][col] == EMPTY


def get_next_open_row(board, col):
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == EMPTY:
            return r


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def winning_move(board, piece):

    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c+i] == piece for i in range(4)):
                return True

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r+i][c] == piece for i in range(4)):
                return True

    # Positive Diagonal
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return True

    # Negative Diagonal
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r-i][c+i] == piece for i in range(4)):
                return True

    return False


def valid_locations(board):
    return [c for c in range(COLS) if is_valid_location(board, c)]


def evaluate_window(window, piece):

    score = 0
    opp = PLAYER if piece == AI else AI

    if window.count(piece) == 4:
        score += 100

    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5

    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score


def score_position(board, piece):

    score = 0

    # Center column preference
    center = [board[r][COLS // 2] for r in range(ROWS)]
    score += center.count(piece) * 3

    # Horizontal
    for r in range(ROWS):
        row = board[r]
        for c in range(COLS - 3):
            score += evaluate_window(row[c:c+4], piece)

    # Vertical
    for c in range(COLS):
        col = [board[r][c] for r in range(ROWS)]
        for r in range(ROWS - 3):
            score += evaluate_window(col[r:r+4], piece)

    # Positive Diagonal
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Negative Diagonal
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score


def terminal(board):
    return winning_move(board, PLAYER) or winning_move(board, AI) or len(valid_locations(board)) == 0


def minimax(board, depth, alpha, beta, maximizing):

    if depth == 0 or terminal(board):

        if winning_move(board, AI):
            return (1000000, None)

        elif winning_move(board, PLAYER):
            return (-1000000, None)

        else:
            return (score_position(board, AI), None)

    if maximizing:

        value = -math.inf
        best_col = random.choice(valid_locations(board))

        for col in valid_locations(board):

            temp = copy.deepcopy(board)
            row = get_next_open_row(temp, col)
            drop_piece(temp, row, col, AI)

            score, _ = minimax(temp, depth - 1, alpha, beta, False)

            if score > value:
                value = score
                best_col = col

            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return value, best_col

    else:

        value = math.inf
        best_col = random.choice(valid_locations(board))

        for col in valid_locations(board):

            temp = copy.deepcopy(board)
            row = get_next_open_row(temp, col)
            drop_piece(temp, row, col, PLAYER)

            score, _ = minimax(temp, depth - 1, alpha, beta, True)

            if score < value:
                value = score
                best_col = col

            beta = min(beta, value)

            if alpha >= beta:
                break

        return value, best_col


board = create_board()

print("CONNECT FOUR")
print("You = 1")
print("Computer = 2")

while True:

    print_board(board)

    col = int(input("Enter column (0-6): "))

    if col < 0 or col > 6:
        print("Invalid Column")
        continue

    if not is_valid_location(board, col):
        print("Column Full")
        continue

    row = get_next_open_row(board, col)
    drop_piece(board, row, col, PLAYER)

    if winning_move(board, PLAYER):
        print_board(board)
        print("You Win!")
        break

    if len(valid_locations(board)) == 0:
        print("Draw")
        break

    _, ai_col = minimax(board, 4, -math.inf, math.inf, True)

    row = get_next_open_row(board, ai_col)
    drop_piece(board, row, ai_col, AI)

    print("Computer played column:", ai_col)

    if winning_move(board, AI):
        print_board(board)
        print("Computer Wins!")
        break