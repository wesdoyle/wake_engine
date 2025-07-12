#!/usr/bin/env python3
"""
Wake Chess Engine - UCI Entry Point

Standalone UCI implementation for packaging with PyInstaller.
"""

import sys
import os
import time

def send_response(message: str):
    """Send UCI response with proper flushing"""
    print(message, flush=True)

def move_to_uci_string(move) -> str:
    """Convert Move object to UCI string format"""
    try:
        from_square = f"{chr(97 + move.from_sq % 8)}{move.from_sq // 8 + 1}"
        to_square = f"{chr(97 + move.to_sq % 8)}{move.to_sq // 8 + 1}"
        move_str = from_square + to_square
        
        if move.is_promotion:
            promotion_map = {1: 'q', 2: 'r', 3: 'b', 4: 'n'}
            if move.promote_to in promotion_map:
                move_str += promotion_map[move.promote_to]
        
        return move_str
    except:
        return "e2e4"

def make_move_from_string(position, move_parser, move_str: str) -> bool:
    """Make a move from UCI string notation"""
    try:
        uci_move = move_parser.parse_input(move_str)
        if uci_move.is_valid and uci_move.is_move:
            piece = position.get_piece_on_square(uci_move.move.from_sq)
            if piece:
                uci_move.move.piece = piece
                position.make_move(uci_move.move)
                return True
    except:
        pass
    return False

def perform_search(position, ai_engine, depth: int):
    """Perform search with progressive output"""
    try:
        best_move = None
        total_nodes = 0
        search_start = time.time()
        
        # Iterative deepening
        for current_depth in range(1, min(depth + 1, 6)):
            result = ai_engine.get_best_move(position, depth=current_depth)
            
            if result and result.best_move:
                best_move = result.best_move
                total_nodes += result.nodes_searched
                elapsed_time = int((time.time() - search_start) * 1000)
                nps = int(total_nodes / (elapsed_time / 1000)) if elapsed_time > 0 else 0
                
                pv_str = move_to_uci_string(best_move)
                send_response(f"info depth {current_depth} score cp {result.evaluation} nodes {total_nodes} time {elapsed_time} nps {nps} pv {pv_str}")
            
            # Limit search time for responsiveness
            if elapsed_time > 10000:  # 10 seconds max
                break
        
        # Send final result
        if best_move:
            send_response(f"bestmove {move_to_uci_string(best_move)}")
        else:
            send_response("bestmove (none)")
            
    except Exception as e:
        send_response(f"info string Search error: {e}")
        send_response("bestmove (none)")

def main():
    """Main UCI loop"""
    try:
        # Import Wake engine components
        from wake.position import Position
        from wake.ai import AIEngine
        from wake.uci_input_parser import UciInputParser
        
        # Initialize components
        position = Position()
        ai_engine = AIEngine(default_depth=4)
        move_parser = UciInputParser()
        
    except Exception as e:
        send_response(f"info string Fatal error initializing engine: {e}")
        send_response("id name Wake Chess Engine (Error)")
        send_response("id author Wake Development Team")
        send_response("uciok")
        return
    
    # Main UCI loop
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            command = line.strip()
            if not command:
                continue
                
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == "uci":
                send_response("id name Wake Chess Engine")
                send_response("id author Wake Development Team")
                send_response("option name Hash type spin default 16 min 1 max 128")
                send_response("option name Threads type spin default 1 min 1 max 8")
                send_response("option name Depth type spin default 4 min 1 max 10")
                send_response("uciok")
                
            elif cmd == "isready":
                send_response("readyok")
                
            elif cmd == "ucinewgame":
                position = Position()
                
            elif cmd == "position":
                if len(parts) > 1:
                    try:
                        if parts[1] == "startpos":
                            position = Position()
                            
                            # Handle moves if present
                            if len(parts) > 2 and parts[2] == "moves":
                                for move_str in parts[3:]:
                                    make_move_from_string(position, move_parser, move_str)
                    except Exception as e:
                        # Reset to startpos on any error
                        position = Position()
                        send_response(f"info string Position error: {e}")
                
            elif cmd == "go":
                # Parse depth
                depth = 4
                for i in range(1, len(parts)):
                    if parts[i] == "depth" and i + 1 < len(parts):
                        try:
                            depth = int(parts[i + 1])
                            depth = max(1, min(depth, 8))
                        except ValueError:
                            pass
                    elif parts[i] == "infinite":
                        depth = 6
                
                # Perform search
                perform_search(position, ai_engine, depth)
                    
            elif cmd == "stop":
                send_response("bestmove (none)")
                    
            elif cmd == "quit":
                break
                
            elif cmd == "debug":
                if len(parts) > 1:
                    if parts[1].lower() == "on":
                        send_response("info string Debug mode enabled")
                    elif parts[1].lower() == "off":
                        send_response("info string Debug mode disabled")
                        
            elif cmd == "setoption":
                if len(parts) >= 4 and parts[1] == "name":
                    option_name = parts[2]
                    if len(parts) >= 5 and parts[3] == "value":
                        option_value = parts[4]
                        send_response(f"info string Set {option_name} to {option_value}")
                        
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            send_response(f"info string Error: {e}")
            continue

if __name__ == "__main__":
    main() 