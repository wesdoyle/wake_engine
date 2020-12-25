# Wake

<img src="./wake.png" width="220px"></img>

Livestream series in progress, but it starts here: https://www.youtube.com/watch?v=1QotIA4_jb4

Wake is a Python (numPy)-based chess engine.  It is currently a work in progress.

## Using the Engine

The current version of Wake is run using Python 3.x from the terminal.

- Clone the directory

- `pip install -r requirements.txt` (this installs the single dependency, `numPy`)

- `cd wake`

- `python3 wake.py`

You will be presented with an output of the board in the shell:

```
Wake Engine [0.1.0] running using interface mode: [uci]
White to move:
8  ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
7  ♟︎ ♟︎ ♟︎ ♟︎ ♟︎ ♟︎ ♟︎ ♟︎
6  ░ ░ ░ ░ ░ ░ ░ ░
5  ░ ░ ░ ░ ░ ░ ░ ░
4  ░ ░ ░ ░ ░ ░ ░ ░
3  ░ ░ ░ ░ ░ ░ ░ ░
2  ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
1  ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
   A B C D E F G H
```

To make a move, use UCI-style move inputs, e.g: `e2e4`.  This will update the
state of the game.

```
Black to move:
8  ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
7  ♟︎ ♟︎ ♟︎ ♟︎ ♟︎ ♟︎ ♟︎ ♟︎
6  ░ ░ ░ ░ ░ ░ ░ ░
5  ░ ░ ░ ░ ░ ░ ░ ░
4  ░ ░ ░ ░ ♙ ░ ░ ░
3  ░ ░ ░ ░ ░ ░ ░ ░
2  ♙ ♙ ♙ ♙ ░ ♙ ♙ ♙
1  ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
   A B C D E F G H
```

## Other Commands

The engine code is currently under development, using the UCI protocol to define valid inputs.

The only valid inputs are currently:

- Any valid move (e.g. `g1f3` for a normal move, a7a8q` for promotion)
- `quit` to kill the engine and terminal input processes.

## Contributing

The engine code will be open for general contribution from the after v1 is complete.

## Roadmap / TODO

- Board representation testing
  - Board representation is in alpha.  A full suite of unit tests still needs to be written to validate behavior.
  - Stalemate, 50 move rule, 3-fold repetition need to be implemented

- Search and Evaluation
  - Crude material evaluation is an attribute of the `Position` class
  - Proper evaluation metrics and implementation TBD

- Rewrite
  - The project is effectively an experiment in building a chess engine from scratch.  Ultimately, the Python code serves as a prototype for a v1 rewrite in Rust.

