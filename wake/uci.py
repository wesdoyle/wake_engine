"""
UCI (Universal Chess Interface) implementation for Wake Chess Engine

This module provides a complete UCI-compliant interface that allows the Wake engine
to communicate with chess GUIs like scidvsmac, Arena, ChessBase, etc.
"""

import sys
import threading
from typing import Optional, List
from wake.constants import Color, UciCommand, UciResponse
from wake.position import Position
from wake.fen import generate_fen, parse_fen
from wake.uci_input_parser import UciInputParser
from wake.ai import AIEngine


class UCIInterface:
    """
    UCI Protocol Implementation for Wake Chess Engine
    
    Handles all UCI commands and provides proper responses according to the
    UCI protocol specification.
    """
    
    def __init__(self):
        self.position = Position()
        self.ai_engine = AIEngine(default_depth=4)
        self.move_parser = UciInputParser()
        self.searching = False
        self.debug_mode = False
        self.engine_name = "Wake Chess Engine"
        self.engine_version = "0.1.0"
        self.author = "Wake Development Team"
        
        # UCI options (can be set by GUI)
        self.options = {
            'Hash': 16,  # Hash table size in MB
            'Threads': 1,  # Number of search threads
            'Depth': 4,  # Search depth
        }
        
    def run(self):
        """Main UCI loop - reads commands from stdin and responds accordingly"""
        try:
            while True:
                try:
                    # Use sys.stdin.readline() instead of input() for better GUI compatibility
                    line = sys.stdin.readline()
                    if not line:  # EOF reached
                        break
                    
                    command = line.strip()
                    if not command:
                        continue
                        
                    self.handle_command(command)
                    
                except (EOFError, BrokenPipeError):
                    # GUI disconnected
                    break
                except Exception as e:
                    if self.debug_mode:
                        self.send_info(f"Error reading input: {e}")
                    continue
                    
        except KeyboardInterrupt:
            pass
    
    def handle_command(self, command: str):
        """Process a UCI command and send appropriate response"""
        parts = command.split()
        if not parts:
            return
            
        cmd = parts[0].lower()
        
        if cmd == "uci":
            self.handle_uci()
        elif cmd == "isready":
            self.handle_isready()
        elif cmd == "ucinewgame":
            self.handle_ucinewgame()
        elif cmd == "position":
            self.handle_position(parts[1:])
        elif cmd == "go":
            self.handle_go(parts[1:])
        elif cmd == "stop":
            self.handle_stop()
        elif cmd == "quit":
            self.handle_quit()
        elif cmd == "debug":
            self.handle_debug(parts[1:])
        elif cmd == "setoption":
            self.handle_setoption(parts[1:])
        else:
            # Unknown command - UCI protocol says to ignore
            if self.debug_mode:
                self.send_info(f"Unknown command: {command}")
    
    def handle_uci(self):
        """Handle 'uci' command - send engine identification"""
        self.send_line(f"id name {self.engine_name} {self.engine_version}")
        self.send_line(f"id author {self.author}")
        
        # Send supported options
        self.send_line("option name Hash type spin default 16 min 1 max 128")
        self.send_line("option name Threads type spin default 1 min 1 max 1")
        self.send_line("option name Depth type spin default 4 min 1 max 10")
        
        self.send_line("uciok")
    
    def handle_isready(self):
        """Handle 'isready' command - confirm engine is ready"""
        self.send_line("readyok")
    
    def handle_ucinewgame(self):
        """Handle 'ucinewgame' command - prepare for new game"""
        self.position = Position()
        self.ai_engine.reset_statistics()
        # Clear any cached data if we had any
        
    def handle_position(self, args: List[str]):
        """Handle 'position' command - set up board position"""
        if not args:
            return
            
        if args[0] == "startpos":
            # Standard starting position
            self.position = Position()
            move_start_idx = 1
        elif args[0] == "fen":
            # Position from FEN string
            if len(args) < 7:  # FEN has 6 parts minimum
                return
            fen = " ".join(args[1:7])
            try:
                self.position = parse_fen(fen)
            except Exception as e:
                if self.debug_mode:
                    self.send_info(f"Invalid FEN: {e}")
                return
            move_start_idx = 7
        else:
            return
            
        # Handle moves after position setup
        if len(args) > move_start_idx and args[move_start_idx] == "moves":
            for move_str in args[move_start_idx + 1:]:
                self.make_move_from_string(move_str)
    
    def handle_go(self, args: List[str]):
        """Handle 'go' command - start searching for best move"""
        if self.searching:
            return
            
        # Parse go command parameters
        depth = None
        movetime = None
        wtime = None
        btime = None
        winc = None
        binc = None
        infinite = False
        
        i = 0
        while i < len(args):
            if args[i] == "depth" and i + 1 < len(args):
                depth = int(args[i + 1])
                i += 2
            elif args[i] == "movetime" and i + 1 < len(args):
                movetime = int(args[i + 1]) / 1000.0  # Convert ms to seconds
                i += 2
            elif args[i] == "wtime" and i + 1 < len(args):
                wtime = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "btime" and i + 1 < len(args):
                btime = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "winc" and i + 1 < len(args):
                winc = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "binc" and i + 1 < len(args):
                binc = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "infinite":
                infinite = True
                i += 1
            else:
                i += 1
        
        # Use depth from go command or default
        if depth is None:
            depth = self.options['Depth']
        
        # Search synchronously for better compatibility
        self.searching = True
        self.search_and_respond(depth, movetime)
    
    def search_and_respond(self, depth: int, movetime: Optional[float] = None):
        """Perform search and send best move response"""
        try:
            # Perform iterative deepening search to show progress
            best_move = None
            best_eval = 0
            total_nodes = 0
            
            for current_depth in range(1, depth + 1):
                if not self.searching:
                    break
                    
                # Perform search at current depth
                result = self.ai_engine.get_best_move(
                    self.position, 
                    depth=current_depth, 
                    time_limit=movetime
                )
                
                if not result or not result.best_move:
                    break
                    
                best_move = result.best_move
                best_eval = result.evaluation
                total_nodes += result.nodes_searched
                
                # Send progressive search info (like other engines)
                time_ms = int(result.time_taken * 1000)
                nps = int(result.nodes_searched / result.time_taken) if result.time_taken > 0 else 0
                
                # Send depth info
                self.send_line(f"info depth {current_depth}")
                
                # Send main search line with PV (principal variation)
                pv_line = self.move_to_uci_string(best_move)
                self.send_line(f"info score cp {best_eval} depth {current_depth} nodes {total_nodes} time {time_ms} pv {pv_line}")
                
                # Send additional stats
                self.send_line(f"info nps {nps}")
            
            # Send final best move
            if best_move:
                best_move_str = self.move_to_uci_string(best_move)
                self.send_line(f"bestmove {best_move_str}")
            else:
                self.send_line("bestmove (none)")
                
        except Exception as e:
            if self.debug_mode:
                self.send_info(f"Search error: {e}")
            # Always send a bestmove response, even on error
            self.send_line("bestmove (none)")
        finally:
            self.searching = False
    
    def handle_stop(self):
        """Handle 'stop' command - stop current search"""
        self.searching = False
        # In a more advanced implementation, we'd interrupt the search thread
    
    def handle_quit(self):
        """Handle 'quit' command - exit engine"""
        sys.exit(0)
    
    def handle_debug(self, args: List[str]):
        """Handle 'debug' command - enable/disable debug mode"""
        if args and args[0].lower() == "on":
            self.debug_mode = True
            self.send_info("Debug mode enabled")
        elif args and args[0].lower() == "off":
            self.debug_mode = False
    
    def handle_setoption(self, args: List[str]):
        """Handle 'setoption' command - set engine options"""
        if len(args) < 4 or args[0] != "name":
            return
            
        option_name = args[1]
        if len(args) >= 4 and args[2] == "value":
            option_value = args[3]
            
            if option_name == "Hash":
                try:
                    self.options['Hash'] = int(option_value)
                except ValueError:
                    pass
            elif option_name == "Threads":
                try:
                    self.options['Threads'] = int(option_value)
                except ValueError:
                    pass
            elif option_name == "Depth":
                try:
                    depth = int(option_value)
                    if 1 <= depth <= 10:
                        self.options['Depth'] = depth
                        self.ai_engine.set_strength(depth)
                except ValueError:
                    pass
    
    def make_move_from_string(self, move_str: str):
        """Make a move from UCI string notation"""
        uci_move = self.move_parser.parse_input(move_str)
        if uci_move.is_valid and uci_move.is_move:
            # Get piece on source square
            piece = self.position.get_piece_on_square(uci_move.move.from_sq)
            if piece:
                uci_move.move.piece = piece
                # Color is automatically determined from piece in Move class
                
                # Make the move
                result = self.position.make_move(uci_move.move)
                if result.is_illegal_move and self.debug_mode:
                    self.send_info(f"Illegal move: {move_str}")
    
    def move_to_uci_string(self, move) -> str:
        """Convert Move object to UCI string format"""
        from_square = f"{chr(97 + move.from_sq % 8)}{move.from_sq // 8 + 1}"
        to_square = f"{chr(97 + move.to_sq % 8)}{move.to_sq // 8 + 1}"
        
        move_str = from_square + to_square
        
        # Add promotion piece if needed
        if move.is_promotion:
            promotion_map = {
                1: 'q',  # Queen
                2: 'r',  # Rook  
                3: 'b',  # Bishop
                4: 'n',  # Knight
            }
            if move.promote_to in promotion_map:
                move_str += promotion_map[move.promote_to]
        
        return move_str
    
    def send_line(self, message: str):
        """Send a line to the GUI"""
        print(message, flush=True)
    
    def send_info(self, message: str):
        """Send info message to GUI"""
        print(f"info string {message}", flush=True)


def main():
    """Main entry point for UCI interface"""
    uci = UCIInterface()
    uci.run()


if __name__ == "__main__":
    main() 