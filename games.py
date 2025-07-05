"""Games or Adversarial Search (Chapter 5)"""

import copy
import random
from collections import namedtuple
import numpy as np
import time

GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

def gen_state(move = '(1, 1)', to_move='X', x_positions=[], o_positions=[], h=3, v=3):
    """
        move = the move that has lead to this state,
        to_move=Whose turn is to move
        x_position=positions on board occupied by X player,
        o_position=positions on board occupied by O player,
        (optionally) number of rows, columns and how many consecutive X's or O's required to win,
    """
    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, move=move, utility=0, board=board, moves=moves)



# ______________________________________________________________________________
# MinMax Search
def minmax(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax:
    # to be implemented by students
    print("your code goes here 5pt")
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)), default=None)


def minmax_cutoff(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the cutoff depth. At that level use evaluation func."""
    print("your code goes here 10pt")

    player = game.to_move(state)

    def max_value(state, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if depth > game.d:
            return game.eval1(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), depth + 1))
        return v

    def min_value(state, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if depth > game.d:
            return game.eval1(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), depth + 1))
        return v

    # Body of minmax:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a), 1), default=None)

def minmax_player(game, state):
    """uses minmax or minmax with cutoff depth, for AI player"""
    print("Your code goes here -5pt")
    """Use a method to speed up at the start to avoid search down a deep tree with not much outcome."""

    # use random player 3 moves before starting min max
    # for sizes 4x4 and 5x5 since my the fourth move the bot needs to
    # start blocking the player
    if (len(state.moves) >= game.size*game.size-3) and game.size > 3:
        return random_player(game, state)

    if game.timer < 0:
        game.d = -1
        return minmax(game, state)

    start = time.perf_counter()
    end = start + game.timer
    """use the above timer to implement iterative deepening loop bellow, using minmax_cutoff(), controlled by the timer"""
    move = None
    print("Your code goes here -5pt")

    depth = 1
    while time.perf_counter() < end:
        game.d = depth
        move = minmax_cutoff(game, state)
        depth += 1

    print("minmax_player: iterative deepening to depth: ", game.d)
    return move



# ______________________________________________________________________________


def alpha_beta(game, state):
    """Search game to determine best action; use alpha-beta pruning.
     this version searches all the way to the leaves."""
    player = game.to_move(state)


    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    alpha = -np.inf
    beta = np.inf
    best_action = max(game.actions(state), key=lambda a: min_value(game.result(state, a), alpha, beta), default=None)
    print("alpha_beta: Your code goes here -15pt")


    return best_action


def alpha_beta_cutoff(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    player = game.to_move(state)

    def max_value(state, alpha, beta, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if depth == game.d:
            return game.eval1(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            # v = max(v, min_value(game.result(state, a), alpha, beta, time.perf_counter()))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if depth == game.d:
            return game.eval1(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            # v = min(v, max_value(game.result(state, a), alpha, beta, time.perf_counter()))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    alpha = -np.inf
    beta = np.inf
    best_action = max(game.actions(state), key=lambda a: min_value(game.result(state, a), alpha, beta, 1), default=None)
    print("Your code goes here -15pt")


    return best_action


def alpha_beta_player(game, state):
    """uses alphaBeta prunning with minmax, or with cutoff version, for AI player"""
    print("Your code goes here -5pt")
    """Use a method to speed up at the start to avoid search down a long tree with not much outcome.
    Hint: for speedup use random_player for start of the game when you see search time is too long"""

    # use random player 3 moves before starting min max
    # for sizes 4x4 and 5x5 since my the fourth move the bot needs to
    # start blocking the player
    if (len(state.moves) >= game.size*game.size-3) and game.size > 3:
        return random_player(game, state)

    if( game.timer < 0):
        game.d = -1
        return alpha_beta(game, state)

    start = time.perf_counter()
    end = start + game.timer
    """use the above timer to implement iterative deepening using alpha_beta_cutoff() version"""
    move = None
    
    print("Your code goes here -5pt")

    depth = 1
    while time.perf_counter() < end:
        game.d = depth
        move = alpha_beta_cutoff(game, state)
        depth += 1

    print("iterative deepening to depth: ", game.d)

    return move


def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None

# ______________________________________________________________________________
# base class for Games

class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))

class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to_move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'.
    depth = -1 means max search tree depth to be used."""

    def __init__(self, size=3, k=3, t=-1):
        self.size = size
        if k <= 0:
            self.k = size
        else:
            self.k = k
        self.d = -1 # d is cutoff depth. Default is -1 meaning no depth limit. It is controlled usually by timer
        self.maxDepth = size * size # max depth possible is width X height of the board
        self.timer = t #timer  in seconds for opponent's search time limit. -1 means unlimited
        moves = [(x, y) for x in range(1, size + 1)
                 for y in range(1, size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def reset(self):
        moves = [(x, y) for x in range(1, self.size + 1)
                 for y in range(1, self.size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    @staticmethod
    def switchPlayer(player):
        assert(player == 'X' or player == 'O')
        return 'O' if player == 'X' else 'X'

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        try:
            moves = list(state.moves)
            moves.remove(move)
        except (ValueError, IndexError, TypeError) as e:
            print("exception: ", e)

        return GameState(to_move=self.switchPlayer(state.to_move), move=move,
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or lost or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(0, self.size):
            for y in range(1, self.size + 1):
                print(board.get((self.size - x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If player wins with this move, return k if player is 'X' and -k if 'O' else return 0."""
        if (self.k_in_row(board, move, player, (0, 1), self.k) or
                self.k_in_row(board, move, player, (1, 0), self.k) or
                self.k_in_row(board, move, player, (1, -1), self.k) or
                self.k_in_row(board, move, player, (1, 1), self.k)):
            return self.k if player == 'X' else -self.k
        else:
            return 0
        
    # evaluation function, version 1
    def eval1(self, state):
        """design and implement evaluation function for state.
        Some ideas: 1-use the number of k-1 matches for X and O For this you can use function possibleKComplete().
            : 2- expand it for all k matches
            : 3- include double matches where one move can generate 2 matches.
            """
        
        """ computes number of (k-1) completed matches. This means number of row or columns or diagonals 
        which include player position and in which k-1 spots are occuppied by player.
        """
        def possiblekComplete(move, board, player, k):
            """if move can complete a line of count items, return 1 for 'X' player and -1 for 'O' player"""
            match = self.k_in_row(board, move, player, (0, 1), k)
            match = match + self.k_in_row(board, move, player, (1, 0), k)
            match = match + self.k_in_row(board, move, player, (1, -1), k)
            match = match + self.k_in_row(board, move, player, (1, 1), k)
            return match

        print("eval1")

        # Maybe to accelerate, return 0 if number of pieces on board is less than half of board size:
        if len(state.moves) <= self.k / 2:
           return 0


        print("Your code goes here 15pt.")


        player = state.to_move
        opponent = 'X' if player == 'O' else 'O'

        score = 0

        win_score = 1000
        block_score = 100
        align_score = 10

        # Iterate through the moves available on the board
        for move in state.moves:
            # Check if the player can complete a line with k-1 moves
            player_potential = possiblekComplete(move, state.board, player, self.k - 1)
            opponent_potential = possiblekComplete(move, state.board, opponent, self.k - 1)

            score += player_potential * align_score

            # Subtract score for opponent's potential to complete
            score -= opponent_potential * block_score

            # Check if the player or opponent can win with this move
            if possiblekComplete(move, state.board, player, self.k) > 0:
                return win_score
            if possiblekComplete(move, state.board, opponent, self.k) > 0:
                return -win_score

        return score


    #@staticmethod
    def k_in_row(self, board, pos, player, dir, k):
        """Return true if there is a line of k cells in direction dir including position pos on board for player."""
        (delta_x, delta_y) = dir
        x, y = pos
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = pos
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= k

