import asyncio
import sys

from wake.bitboard_helpers import pprint_pieces
from wake.constants import Color
from wake.position import Position
from wake.uci_input_parser import UciInputParser


async def connect_stdin_stdout():
    event_loop = asyncio.get_event_loop()
    stream_reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(stream_reader)
    await event_loop.connect_read_pipe(lambda: protocol, sys.stdin)
    w_transport, w_protocol = await event_loop.connect_write_pipe(asyncio.streams.FlowControlMixin, sys.stdout)
    stream_writer = asyncio.StreamWriter(w_transport, w_protocol, stream_reader, event_loop)
    return stream_reader, stream_writer


class Game:

    def __init__(self, mode):
        self.mode = mode.strip().lower()
        self.history = []  # Stack of FENs (TODO: consider stacking position states)
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
            await self.run_uci_mode()
        else:
            print("That's not a valid mode.")
            self.run()

    async def try_parse_move(self, move):
        engine_input = await self.parser.parse_input(move)
        if not engine_input.is_valid:
            print("Invalid input")
            return None
        if engine_input.is_move:
            move_piece = self.position.get_piece_on_square(engine_input.move.from_sq)
            if not move_piece:
                print("Invalid input")
                return None
            engine_input.move.piece = move_piece
            return engine_input.move

    async def run_uci_mode(self):
        reader, writer = await connect_stdin_stdout()

        while not self.is_over:
            print(f"{self.color_to_move[self.position.color_to_move]} to move:")

            uci_input = await reader.readuntil()

            move = await self.try_parse_move(uci_input)

            if not move:
                print("Invalid move format")
                continue

            move_result = self.position.make_move(move)

            if move_result.is_checkmate:
                print("Checkmate")
                self.score[self.position.color_to_move] = 1
                self.is_over = True

            if move_result.is_stalemate:
                print("Stalemate")
                self.score = [0.5, 0.5]
                self.is_over = True

            self.history.append(move_result.fen)
            pprint_pieces(self.position.piece_map)


import multiprocessing as mp
import os


def engine_proc(queue, mode):

    print("Using input mode:", mode)

    game = Game(mode)
    # game.run()

    while True:
        if not queue.empty():
            msg = queue.get()
            print("<< Engine Got new message", msg.strip())


def reader_proc(queue, fileno):
    sys.stdin = os.fdopen(fileno)
    while True:
        input_move = sys.stdin.readline()
        queue.put(input_move)
        print(f">> Reader sending {input_move.strip()} to work Queue()")


if __name__ == "__main__":
    mode = input("Welcome to Wake. Choose an input mode [UCI]:")

    inp_queue = mp.Queue()

    engine_process = mp.Process(target=engine_proc, args=(inp_queue, mode))
    engine_process.daemon = True
    engine_process.start()
    fn = sys.stdin.fileno()

    reader_process = mp.Process(target=reader_proc, args=(inp_queue,fn))
    reader_process.daemon = False
    reader_process.start()

    print("Wake Engine [0.1.0] started")

    engine_process.join()
    reader_process.join()

    print("Goodbye")
