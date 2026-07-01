import json
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np
import psutil
import time
import os
import signal

def slice_to_viz_json(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    blocks = data["blocks"]
    palette = data["palette"].tolist()

    # FIXED: Extracting metadata safely from numpy object
    meta_raw = data["meta"]
    if hasattr(meta_raw, "item"):
        meta_raw = meta_raw.item()  # Extracts string from numpy array
    metadata = json.loads(meta_raw)

    # Map Minecraft blocks to hex colors for the HTML visualizer
    color_map = {
        # Environment
        "minecraft:air": "#00000000",
        "minecraft:grass_block": "#5D8A3C",
        "minecraft:dirt": "#866043",
        "minecraft:water": "#2255CC",
        "minecraft:stone": "#B7B6B6",
        "minecraft:coarse_dirt": "#B49C5A",

        # Infrastructure
        "minecraft:cobblestone": "#FFFFFF",      # Main roads
        "minecraft:oak_planks": "#B8902A",       # Bridges
        "minecraft:yellow_concrete": "#FAFD97",  # Interior paths
        "minecraft:dirt_path": "#74580A",

        # Settlement Features
        "minecraft:red_wool": "#B1635F",         # Perimeter walls
        "minecraft:iron_block": "#D5D5D5",       # Cardinal doors
        "minecraft:emerald_block": "#2ECC71",    # Town center marker
        "minecraft:gold_block": "#F1C40F",        # Buildable cell markers
        "minecraft:lime_concrete": "#41A75B",
        "minecraft:terracotta": "#87E9FE"
    }
    return {
        "origin": data["origin"].tolist(),
        "size": list(blocks.shape),
        "palette": palette,
        "color_map": color_map,
        "blocks": base64.b64encode(blocks.astype(np.uint16).tobytes()).decode(),
        "metadata": metadata
    }

class Handler(BaseHTTPRequestHandler):
    json_data = b""
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(Path("voxel_visualizer.html").read_bytes())
        elif self.path == "/slice.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(self.json_data)

def serve(npz_path, port=8765):
    Handler.json_data = json.dumps(slice_to_viz_json(Path(npz_path))).encode()
    print(f"Server started at http://localhost:{port} using {npz_path}")
    HTTPServer(("localhost", port), Handler).serve_forever()


def kill_process_on_port(port):
    """Finds and terminates the process using the specified port."""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conns in proc.connections(kind='inet'):
                if conns.laddr.port == port:
                    print(f"Terminating ghost process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}...")
                    proc.send_signal(signal.SIGTERM) # Gracious kill
                    time.sleep(0.5)
                    if proc.is_running():
                        proc.kill() # Force kill if necessary
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def force_serve(npz_path, port=8765):
    """Kills existing port owner and starts the server."""
    kill_process_on_port(port)
    # Give the OS a tiny window to fully release the socket
    time.sleep(0.2)
    print(f"Launching fresh server on port {port}...")
    serve(npz_path, port=port)

if __name__ == "__main__":
    serve("data/settlement_viz.npz")