# Generalized Tic-Tac-Toe with Adversarial Search

A Python project implementing a flexible n x n Tic-Tac-Toe game with powerful adversarial search AI agents: **MinMax**, **AlphaBeta Pruning**, and **Monte Carlo Tree Search (MCTS)**. Features a GUI for interactive play and a timer-based AI cutoff mechanism. Designed for both playability and as a showcase of classical and modern AI search techniques.

---

## Features

- **Generalized Game Board:** Play on any size (3x3, 4x4, 5x5, etc.)
- **Multiple AI Agents:**
  - **MinMax** (full depth or with depth cutoff)
  - **AlphaBeta Pruning** (full depth or with depth cutoff)
  - **Monte Carlo Tree Search (MCTS)**
  - **Random Player** (for benchmarking)
- **Timer Control:** Set a search time limit for AI moves, enabling playable larger boards.
- **Custom Evaluation Function:** Heuristic scoring to guide AI at depth limits.
- **Tkinter GUI:** Intuitive board interface with mode selection.
- **Playable Human vs. AI:** Choose any algorithm and board size.
- **Modular & Extensible:** Clean architecture, easily adaptable for new algorithms or heuristics.

---

## File Structure

```plaintext
.
├── games.py           # Core adversarial search algorithms, evaluation, and game mechanics
├── monteCarlo.py      # Monte Carlo Tree Search (MCTS) implementation
├── tic-tac-toe.py     # Tkinter-based GUI, main game loop
├── README.md          # Project documentation
└── (Other helpers/scripts as needed)
```
# Algorithms Implemented
1. MinMax Search

   - Classical adversarial search, explores all game states to select the optimal move.

   - Depth-cutoff Version: Limits depth and evaluates non-terminal nodes with a heuristic.

2. AlphaBeta Pruning

   - Optimized MinMax with branch pruning to reduce computation.

   - Depth-cutoff Version: Integrates cutoff and evaluation heuristics for large boards.

3. Monte Carlo Tree Search (MCTS)

   - AI agent that uses random simulations ("playouts") to build a game tree, balancing exploration vs. exploitation.

   - UCT (Upper Confidence Bounds for Trees): Guides node selection.

   - Iterative Deepening: Repeats simulations within a time budget to maximize move quality.

# Game Architecture

   - TicTacToe (games.py): Encapsulates the board, rules, and utility computation.

   - GameState: Encodes the state, player turn, board, available moves, etc.

   - Evaluation Function: Smart heuristic in eval1() scoring board states for non-terminal positions.

   - Players: Each AI agent is implemented as a "player" callable.

## Usage
# Requirements

  -  Python 3.x

  - Tkinter (usually included with Python)

  - NumPy

# Running the Game
```sh
python tic-tac-toe.py
```
   - Use the dropdown menu to select the AI type: Random, MinMax, AlphaBeta, MonteCarlo.

   - Set the Timer (in seconds) for AI moves. -1 disables the limit (suitable for small boards).

Command-Line Board Size
```sh
python tic-tac-toe.py 4   # Starts a 4x4 Tic-Tac-Toe game
```
# Evaluation Function

Implemented in ```games.py``` → ```TicTacToe.eval1()```, it quantifies the win potential for both the AI and opponent, rewarding potential wins and penalizing threats. You can extend this to experiment with more advanced heuristics (double threats, blocking forks, etc.).

## How It Works

   - Select Mode: Choose AI (Random/MinMax/AlphaBeta/MCTS).

   - Human Plays as 'X'. On each move, the AI computes and plays as 'O'.

   - Timer: AI respects the per-move timer (for larger boards, keep this 3–5s).

   - Terminal States: Game detects wins, losses, and draws, ending the session.

## Sample: MinMax with Depth Cutoff (Pseudocode)
```python
def minmax_cutoff(game, state):
    def max_value(state, depth):
        if game.terminal_test(state): return game.utility(state, player)
        if depth > max_depth: return evaluation(state)
        v = -inf
        for action in game.actions(state):
            v = max(v, min_value(game.result(state, action), depth+1))
        return v
    # ...

```
## Extending This Project

   - Smarter Heuristics: Implement pattern recognition or machine-learned scoring in eval1().

   - Custom Game Rules: Adapt to Connect-4, Gomoku, or other n-in-a-row games.

  -  RL/Neural Agents: Integrate Reinforcement Learning agents for comparison.

  - Web UI: Swap Tkinter for a web frontend (Flask/Streamlit/React).

  -  Parallel MCTS: Use multi-threading for even stronger MCTS play.

## Why This Project Matters

  -  Educational Value: Demonstrates foundational adversarial search and MCTS in a real, interactive setting.

   - AI Research: Framework for experimenting with and benchmarking classical and modern game AI methods.

   - Portfolio Piece: Clean, extensible code and interactive demo for showcasing AI/ML and software engineering skills.

Credits

   - Assignment prompt: [CMPT 310: Adversarial Search Assignment, Spring 2025]

   - AI methods: Russell & Norvig, "Artificial Intelligence: A Modern Approach"

## License
-  For educational use only. Do not redistribute without permission. If reusing or extending for other coursework, please credit the original source.
