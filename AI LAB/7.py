# Minimax Algorithm

def minimax(depth, nodeIndex, isMax, scores, height):

    if depth == height:
        return scores[nodeIndex]

    if isMax:
        return max(
            minimax(depth + 1, nodeIndex * 2, False, scores, height),
            minimax(depth + 1, nodeIndex * 2 + 1, False, scores, height)
        )
    else:
        return min(
            minimax(depth + 1, nodeIndex * 2, True, scores, height),
            minimax(depth + 1, nodeIndex * 2 + 1, True, scores, height)
        )

# Leaf node values
scores = [3, 5, 2, 9]

height = 2

result = minimax(0, 0, True, scores, height)

print("The optimal value is:", result)