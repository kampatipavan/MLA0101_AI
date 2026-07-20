# Alpha-Beta Pruning Algorithm

MIN = -1000
MAX = 1000

def alphabeta(depth, nodeIndex, maximizingPlayer, values, alpha, beta):

    if depth == 2:
        return values[nodeIndex]

    if maximizingPlayer:

        best = MIN

        for i in range(2):

            val = alphabeta(depth + 1,
                            nodeIndex * 2 + i,
                            False,
                            values,
                            alpha,
                            beta)

            best = max(best, val)
            alpha = max(alpha, best)

            if beta <= alpha:
                break

        return best

    else:

        best = MAX

        for i in range(2):

            val = alphabeta(depth + 1,
                            nodeIndex * 2 + i,
                            True,
                            values,
                            alpha,
                            beta)

            best = min(best, val)
            beta = min(beta, best)

            if beta <= alpha:
                break

        return best


values = [3, 5, 6, 9]

result = alphabeta(0, 0, True, values, MIN, MAX)

print("The optimal value is:", result)