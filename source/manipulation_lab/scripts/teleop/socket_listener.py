"""Socket listener for remote teleoperation."""

import logging
logger = logging.getLogger("ManipulationLab.SocketListener")

from concurrent.futures import thread
import socket
from multiprocessing import Array
import threading

def start_socket_listener(controller_array: Array, port: int = 8888, expected_dims: int = 11):
    def _listener():
        # Initialise socket
        s = socket.socket()

        # Listen on all interfaces
        s.bind(("0.0.0.0", port))

        # Accept only one connection
        s.listen(1)
        
        # Accept inbound connection
        conn, addr = s.accept()
        logger.info(f"Teleop connected from {addr}")

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
                    if len(parts) == expected_dims:
                        for idx, value in enumerate(parts):
                            controller_array[idx] = value
                    else: 
                        logger.warning(f"Received message with unexpected dimensions: {len(parts)}")
                except Exception as e:
                    logger.warning(f"Failed parsing message: {e}")

    # Start thread to execute _listener()
    logger.info("Starting socket listener")
    thread = threading.Thread(target=_listener, daemon=True)
    thread.start()

    return thread