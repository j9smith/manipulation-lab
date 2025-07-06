"""Socket listener for remote teleoperation."""

from concurrent.futures import thread
import socket
from multiprocessing import Array
import threading

def start_socket_listener(action_array: Array, port: int = 8888):
    def _listener():
        # Initialise socket
        s = socket.socket()

        # Listen on all interfaces
        s.bind(("0.0.0.0", port))

        # Accept only one connection
        s.listen(1)
        
        # Accept inbound connection
        conn, addr = s.accept()
        print(f"Teleop connected from {addr}")

        # Initialise byte buffer
        buffer = b""

        while True:
            # Read 1024 bytes at a time
            data = conn.recv(1024)

            if not data:
                continue
            buffer += data

            # While there is a new line in the buffer, decode the message and put it in the queue
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                msg = line.decode().strip()
                
                try:
                    parts = list(map(float, msg.split(',')))
                    if len(parts) == 6:
                        for idx, value in enumerate(parts):
                            action_array[idx] = value
                except Exception as e:
                    print(f"Failed parsing message: {e}")

    # Start thread to execute _listener()
    print("Starting socket listener")
    thread = threading.Thread(target=_listener, daemon=True)
    thread.start()

    return thread