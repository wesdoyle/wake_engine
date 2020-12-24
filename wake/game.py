import sys
import asyncio
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

    def __init__(self):
        self.history = []  # Stack of FENs (TODO: consider stacking position states)
        self.position = Position()
        self.is_over = False
        self.score = [0, 0]
        self.parser = UciInputParser()

        self.color_to_move = {
            Color.WHITE: "White",
            Color.BLACK: "Black",
        }

    async def run(self):
        reader, writer = await connect_stdin_stdout()
        print("Choose input mode (UCI is the only mode available): ")

        res = await reader.readuntil()

        if res:
            # if str(res).strip().lower() == b'uci\n':
            print("OK")
            await self.run_uci_mode()

        else:
            await self.run()

        print(self.score)

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


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        game = Game()
        loop.run_until_complete(game.run())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
