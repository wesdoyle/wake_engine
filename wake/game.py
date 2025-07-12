import multiprocessing as mp
import os
import sys
import time
from os import system, name

from wake.bitboard_helpers import pprint_pieces
from wake.constants import Color, UciCommand
from wake.position import Position
from wake.uci_input_parser import UciInputParser

CURRENT_VERSION = "0.1.0"


def clear():
    # Windows
    if name == "nt":
        _ = system("cls")
    else:
        _ = system("clear")


class Game:

    def __init__(self, interface_mode, queue, sys_queue):
        self.mode = interface_mode.strip().lower()
        self.queue = queue
        self.sys_queue = sys_queue
        self.history = []
        self.position = Position()
        self.is_over = False
        self.score = [0, 0]
        self.parser = UciInputParser()

        self.color_to_move = {
            Color.WHITE: "White",
            Color.BLACK: "Black",
        }

    def run(self):
        if self.mode == "uci":
            self.run_uci_mode()

    def try_parse_move(self, move_str):
        engine_input = self.parser.parse_input(move_str)
        if not engine_input.is_valid:
            return (
                None,
                f"Invalid move format: '{move_str}'. Use format like 'e2e4' or 'a7a8q'.",
            )
        if engine_input.is_move:
            move_piece = self.position.get_piece_on_square(engine_input.move.from_sq)
            if not move_piece:
                return None, f"No piece on square {move_str[:2]}."
            engine_input.move.piece = move_piece
            return engine_input.move, None
        return None, "Unknown error parsing move."

    def run_uci_mode(self):
        sentinel = False
        while True:
            if not sentinel:
                print(
                    f"Wake Engine [{CURRENT_VERSION}] running using interface mode: [{self.mode}]"
                )
                print(f"{self.color_to_move[self.position.color_to_move]} to move:")
                sentinel = True
                pprint_pieces(self.position.piece_map)
            if not self.queue.empty():
                clear()
                msg = self.queue.get().strip()
                move, error_msg = self.try_parse_move(msg)

                if not move:
                    if msg == UciCommand.QUIT:
                        self.sys_queue.put("kill")
                    else:
                        # Invalid move - provide specific feedback and reset display
                        print(error_msg)
                        print(
                            "Please enter a valid move in UCI format (e.g., 'e2e4') or 'quit' to exit."
                        )
                        print()
                        sentinel = False  # Reset to re-display the board
                    continue

                move_result = self.position.make_move(move)

                if move_result.is_illegal_move:
                    print(f"Illegal move: {msg}")
                    print("Please try a different move.")
                    print()
                    sentinel = False  # Reset to re-display the board
                    continue

                if move_result.is_checkmate:
                    print("Checkmate")
                    self.score[self.position.color_to_move] = 1
                    self.is_over = True

                if move_result.is_stalemate:
                    print("Stalemate")
                    self.score = [0.5, 0.5]
                    self.is_over = True

                self.history.append(move_result.fen)
                sentinel = False

            else:
                pass


def engine_proc(queue, sys_queue, interface_mode="uci"):
    clear()
    game = Game(interface_mode, queue, sys_queue)
    game.run()


def reader_proc(queue, fileno):
    sys.stdin = os.fdopen(fileno)
    while True:
        input_move = sys.stdin.readline()
        queue.put(input_move)


if __name__ == "__main__":
    mode = "UCI"
    inp_queue = mp.Queue()
    sys_queue = mp.Queue()

    engine_process = mp.Process(target=engine_proc, args=(inp_queue, sys_queue, mode))
    engine_process.daemon = True
    engine_process.start()
    fn = sys.stdin.fileno()

    reader_process = mp.Process(target=reader_proc, args=(inp_queue, fn))
    reader_process.daemon = False
    reader_process.start()

    while True:
        message = sys_queue.get()
        if message == "kill":
            engine_process.terminate()
            reader_process.terminate()
            time.sleep(0.5)
            if not engine_process.is_alive() or not reader_process.is_alive():
                engine_process.join()
                reader_process.join()
                print("Peace")
                sys.exit(0)
