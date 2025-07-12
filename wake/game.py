import multiprocessing as mp
import os
import sys
import time
from os import system, name

from wake.bitboard_helpers import pprint_pieces
from wake.constants import Color, UciCommand
from wake.position import Position
from wake.uci_input_parser import UciInputParser
from wake.ai import AIEngine

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
        elif self.mode == "human_vs_ai":
            self.run_human_vs_ai_mode()

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

    def setup_human_vs_ai(self):
        """Setup human vs AI game: choose color and AI depth"""
        clear()
        print("Welcome to Wake Chess Engine!")
        print("=" * 50)
        print("Human vs AI Mode")
        print()
        
        # Color selection
        while True:
            color_choice = input("Choose your color (w/white or b/black): ").strip().lower()
            if color_choice in ['w', 'white']:
                self.human_color = Color.WHITE
                self.ai_color = Color.BLACK
                print(f"You are playing as WHITE")
                break
            elif color_choice in ['b', 'black']:
                self.human_color = Color.BLACK  
                self.ai_color = Color.WHITE
                print(f"You are playing as BLACK")
                break
            else:
                print("Invalid choice. Please enter 'w' for white or 'b' for black.")
                
        print()
        
        # AI depth selection
        while True:
            try:
                depth_choice = input("Choose AI search depth (1-3, recommended 2): ").strip()
                depth = int(depth_choice)
                if 1 <= depth <= 3:
                    self.ai_depth = depth
                    print(f"AI will search to depth {depth}")
                    break
                else:
                    print("Depth must be between 1 and 3.")
            except ValueError:
                print("Please enter a valid number (1-3).")
        
        print()
        print("Game starting! Enter moves in UCI format (e.g., 'e2e4')")
        print("Type 'quit' to exit, 'moves' to see legal moves")
        print()
        
        # Initialize AI engine
        self.ai_engine = AIEngine(default_depth=self.ai_depth)
        
        input("Press Enter to begin...")

    def run_human_vs_ai_mode(self):
        """Main human vs AI game loop"""
        self.setup_human_vs_ai()
        
        while not self.is_over:
            clear()
            self.display_game_state()
            
            if self.position.color_to_move == self.human_color:
                self.handle_human_turn()
            else:
                self.handle_ai_turn()
    
    def display_game_state(self):
        """Display current board and game information"""
        print("Wake Chess Engine - Human vs AI")
        print("=" * 50)
        print()
        
        # Show whose turn it is
        current_player = "Your turn" if self.position.color_to_move == self.human_color else "AI thinking..."
        color_name = "White" if self.position.color_to_move == Color.WHITE else "Black"
        print(f"{current_player} ({color_name})")
        print()
        
        # Display board
        pprint_pieces(self.position.piece_map)
        print()
        
        # Show game statistics
        legal_moves = self.position.generate_legal_moves()
        print(f"Legal moves available: {len(legal_moves)}")
        
        # Show position evaluation
        try:
            from wake.ai.evaluation import evaluate_position
            evaluation = evaluate_position(self.position)
            eval_display = f"{evaluation/100:+.2f}"
            print(f"Position evaluation: {eval_display} (White's perspective)")
        except:
            pass
        
        print()
    
    def handle_human_turn(self):
        """Handle human player input"""
        while True:
            move_input = input("Enter your move (or 'quit'/'moves'): ").strip().lower()
            
            if move_input == 'quit':
                print("Thanks for playing!")
                self.is_over = True
                return
                
            if move_input == 'moves':
                self.show_legal_moves()
                continue
                
            # Try to parse and make the move
            move, error_msg = self.try_parse_move(move_input)
            if not move:
                print(f"{error_msg}")
                continue
                
            move_result = self.position.make_move(move)
            
            if move_result.is_illegal_move:
                print(f"Illegal move: {move_input}")
                continue
                
            # Move was successful
            print(f"You played: {move_input}")
            self.check_game_over(move_result)
            break
    
    def handle_ai_turn(self):
        """Handle AI move with debugging output"""
        print("AI is thinking...")
        print()
        
        # Show AI analysis
        legal_moves = self.position.generate_legal_moves()
        print(f"AI analyzing {len(legal_moves)} candidate moves:")
        
        # Show some top candidate moves (first 5)
        print("   Top candidate moves:")
        for i, move in enumerate(legal_moves[:5]):
            move_str = f"{chr(97 + move.from_sq % 8)}{move.from_sq // 8 + 1}"
            move_str += f"{chr(97 + move.to_sq % 8)}{move.to_sq // 8 + 1}"
            capture_note = " (capture)" if move.is_capture else ""
            print(f"   {i+1}. {move_str}{capture_note}")
        
        if len(legal_moves) > 5:
            print(f"   ... and {len(legal_moves) - 5} more")
        print()
        
        # Get AI decision
        print(f"AI searching to depth {self.ai_depth}...")
        result = self.ai_engine.get_best_move(self.position, depth=self.ai_depth)
        
        if not result.best_move:
            print("AI couldn't find a move!")
            self.is_over = True
            return
            
        # Show AI analysis results
        print("AI Analysis Complete:")
        print(f"   Best move: {self.format_move(result.best_move)}")
        print(f"   Evaluation: {result.evaluation/100:+.2f}")
        print(f"   Nodes searched: {result.nodes_searched:,}")
        print(f"   Time taken: {result.time_taken:.3f}s")
        print(f"   Nodes/sec: {result.nodes_searched/result.time_taken:,.0f}")
        print()
        
        # Make the AI move
        move_result = self.position.make_move(result.best_move)
        
        if move_result.is_illegal_move:
            print("AI tried to make an illegal move!")
            self.is_over = True
            return
            
        print(f"AI played: {self.format_move(result.best_move)}")
        self.check_game_over(move_result)
        
        input("\nPress Enter to continue...")
    
    def format_move(self, move):
        """Format a move object as UCI notation"""
        from_square = f"{chr(97 + move.from_sq % 8)}{move.from_sq // 8 + 1}"
        to_square = f"{chr(97 + move.to_sq % 8)}{move.to_sq // 8 + 1}"
        move_str = from_square + to_square
        
        annotations = []
        if move.is_capture:
            annotations.append("capture")
        if move.is_castling:
            annotations.append("castling")
        if move.is_promotion:
            annotations.append("promotion")
            
        if annotations:
            move_str += f" ({', '.join(annotations)})"
            
        return move_str
    
    def show_legal_moves(self):
        """Display all legal moves for the current player"""
        legal_moves = self.position.generate_legal_moves()
        print(f"\nLegal moves ({len(legal_moves)} total):")
        
        moves_per_line = 6
        for i, move in enumerate(legal_moves):
            if i % moves_per_line == 0:
                print("   ", end="")
            
            move_str = self.format_move(move)
            print(f"{move_str:<12}", end="")
            
            if (i + 1) % moves_per_line == 0:
                print()  # New line
        
        if len(legal_moves) % moves_per_line != 0:
            print()  # Final new line if needed
        print()
    
    def check_game_over(self, move_result):
        """Check if the game has ended"""
        if move_result.is_checkmate:
            winner = "You win!" if self.position.color_to_move != self.human_color else "AI wins!"
            print(f"Checkmate! {winner}")
            self.is_over = True
            
        elif move_result.is_stalemate:
            print("Stalemate - Draw!")
            self.is_over = True
            
        elif move_result.is_draw_claim_allowed:
            print("Draw!")
            self.is_over = True

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
                    print("Stalemate (Draw)")
                    self.score = [0.5, 0.5]
                    self.is_over = True

                if move_result.is_draw_claim_allowed:
                    print("Draw")
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


def run_simple_human_vs_ai():
    """Run simple human vs AI mode without multiprocessing"""
    clear()
    game = Game("human_vs_ai", None, None)
    game.run()


if __name__ == "__main__":
    clear()
    print("Welcome to Wake Chess Engine!")
    print("=" * 50)
    print("Choose your interface mode:")
    print("1. Human vs AI (Interactive)")
    print("2. UCI Mode (Engine interface)")
    print()
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            run_simple_human_vs_ai()
            break
        elif choice == "2":
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
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
