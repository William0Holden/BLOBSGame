# William Holden
# Define a BLOBS game in your project file BLOBSGame.py
from games import Game
from games import GameState
from games import alpha_beta_search
from collections import namedtuple
from games import query_player
import random
import numpy as np

GameState = namedtuple('GameState', 'to_move, utility, board, moves, prev_move')

class BLOBS(Game):

    # You will define the following methods:
    def __init__(self, h = 7, v = 7) -> None:
        # Initialize your game here
        self.h = h
        self.v = v
        initial_moves = []
        board = [['.' for _ in range(v)] for _ in range(h)]
        board[0][0] = 'X'
        board[0][v-1] = 'O'
        board[h-1][0] = 'O'
        board[h-1][v-1] = 'X'
        self.initial = GameState(to_move='X', utility=0, board=board, moves=self.find_moves(board, 'X'), prev_move=None)

    # Returns the new state that results from making a move
    def result(self, state, move):
        x, y, nx, ny, action = move  
        new_board = deep_copy_board(state.board)
        player = state.to_move  
        
        if action == "PLACE":  #new tile is placed
            new_board[nx][ny] = player
        elif action == "MOVE":  # tile jumps over another space (more efficient to place adjacent than to move adjacent)
            new_board[x][y] = '.' 
            new_board[nx][ny] = player 
        next_move = 'O' if player == 'X' else 'X'
        return GameState(to_move=next_move, utility=self.utility(new_board, next_move),
                          board=new_board, moves=self.find_moves(new_board, next_move),
                          prev_move=move)
    
    # Returns a list of all possible legal actions from state
    def actions(self, state):
        return state.moves
    
    def find_moves(self, board, type): #finds all possible moves for a given player
        moves = []
        for i in range(self.h):
            for j in range(self.v):
                if board[i][j] == type: #finds all spaces with the given player's color
                    moves.extend(self.find_moves_for_this_space(board, i, j)) #finds all possible moves for that space
        return moves

    def find_moves_for_this_space(self, board, x, y):
        tempMoves = []
        for x_distance in range(-2, 3):
            for y_distance in range(-2, 3):
                if x_distance == 0 and y_distance == 0:
                    continue
                if x + x_distance < 0 or x + x_distance > self.h - 1: #checks if the move is out of bounds
                    continue
                if y + y_distance < 0 or y + y_distance > self.v - 1: #checks if the move is out of bounds
                    continue
                if board[x + x_distance][y + y_distance] == '.':
                    if abs(x_distance) <= 1 and abs(y_distance) <= 1:
                        tempMoves.append((x, y, x + x_distance, y + y_distance, "PLACE")) #if the move is adjacent, it is a place move
                    elif abs(x_distance) > 1 or abs(y_distance) > 1:
                        tempMoves.append((x, y, x + x_distance, y + y_distance, "MOVE")) #if the move is not adjacent, it is a move (hop) move
                    else:
                        tempMoves = None
        return tempMoves
    
    def display(self, state):
        board = state.board
        for x in range(self.h):
            for y in range(self.v):
                print(board[x][y], end=' ')
            print()

    # Is the game finished? Return true if the game is in an end state
    def terminal_test(self, state):
        if state.prev_move is None and not state.moves:
            return True
        return len(state.moves) == 0
        
    # Returns the utility value (evaluation) of the current state for player 'p'
    def utility(self, board, player):
        if isinstance(board, list):
            Xsums = 0
            Osums = 0
            for x in range(self.h - 1):
                for y in range(self.v - 1):
                    if board[x][y] == 'X':
                        Xsums += 1
                    elif board[x][y] == 'O':
                        Osums += 1
            if player == 'X':
                return Xsums - Osums #returns the difference between the number of X's and O' on the board
            else:
                return Osums - Xsums 
        
    def play_game(self, *players):
        state = self.initial
        skips = 0
        while True:
            for player in players:
                move = player(self, state)
                if move is None:
                    skips += 1
                else:
                    skips = 0
                    state = self.result(state, move)
                if skips > 1 or self.terminal_test(state): 
                    self.display(state)
                    return state
                
    def end_game_stats(self, state): #prints the winner and the number of spaces filled by each player
        board = state.board
        xSums = 0
        oSums = 0
        for x in range(self.h):
            for y in range(self.v):
                if board[x][y] == 'X':
                    xSums += 1
                elif board[x][y] == 'O':
                    oSums += 1
        if xSums > oSums:
            print("Player X wins! They have {} spaces filled and Player O has {} spaces filled\n".format(xSums, oSums))
        elif oSums > xSums:
            print("Player O wins! They have {} spaces filled and Player X has {} spaces filled\n".format(oSums, xSums))

    def eval(self, state): # returns the difference between the number of moves the player can make and the number of moves the opponent can make (takes a long time to run)
        player = state.to_move
        opponentMoves = 0
        playerMoves = 0
        for x in range(self.h):
            for y in range(self.v):
                if state.board[x][y] == player:
                    playerMoves += len(self.find_moves_for_this_space(state.board, x, y))
                else:
                    opponentMoves += len(self.find_moves_for_this_space(state.board, x, y))
        return playerMoves - opponentMoves

def minmax_decision_with_cutoff(state, game, depth=3, eval_fn=None, cutoff_test=None):
    eval_fn = eval_fn or game.eval #if eval_fn is not provided, use the game's eval function
    player = game.to_move(state)

    def max_value(state, depth): #returns the max value of the state
        if depth <= 0 or game.terminal_test(state):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), depth - 1))
        return v

    def min_value(state, depth): #returns the min value of the state
        if depth <= 0 or game.terminal_test(state):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), depth - 1))
        return v
    
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a), depth - 1)) #returns the action that results in the max value

def alpha_beta_cutoff_search(state, game, d=3, cutoff_test=None, eval_fn=None):
    player = game.to_move(state)

    # Check if cutoff_test and eval_fn are provided and set default functions
    cutoff_test = cutoff_test or (lambda state, depth: depth >= d or game.terminal_test(state)) #if cutoff_test is not provided, use the default cutoff_test
    eval_fn = eval_fn or (lambda state: game.eval(state)) #if eval_fn is not provided, use the game's eval function

    # Functions used by alpha-beta
    def max_value(state, alpha, beta, depth): #returns the max value of the state
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth): #returns the min value of the state
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state): #returns the action that results in the max value
        v = min_value(game.result(state, a), best_score, beta, depth = 1) 
        if v > best_score:
            best_score = v
            best_action = a
    return best_action


def minmax_player_with_cutoff(game, state): # creates a player that uses minmax with cutoff
    return minmax_decision_with_cutoff(state, game)

def alpha_beta_player(game, state): # creates a player that uses alpha beta pruning
    return alpha_beta_cutoff_search(state, game)

def random_player(game, state): #from games.py
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None

def deep_copy_board(board):
    return [row[:] for row in board]

#main function
def main():
    game = BLOBS(5,5)

    # Random vs Random
    print("Random vs Random")
    state = game.play_game(random_player, random_player)
    game.end_game_stats(state)

    # Random vs MiniMax with Cuttoff
    print("Random vs MiniMax with Cuttoff")
    state = game.play_game(random_player, minmax_player_with_cutoff)
    game.end_game_stats(state)

    # MiniMax with Cuttoff vs AlphaBeta Pruning
    print("AlphaBeta Pruning vs MiniMax with Cuttoff")
    state = game.play_game(alpha_beta_player, minmax_player_with_cutoff)
    game.end_game_stats(state)

if __name__ == "__main__":
    main() 