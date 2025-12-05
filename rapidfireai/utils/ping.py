#!/usr/bin/env python
import argparse
import socket


def ping_server(server: str, port: int, timeout=3):
    """ping server:port"""
    try:
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((server, port))
    except OSError as _:
        return False
    else:
        s.close()
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ping a server port")
    parser.add_argument("server", type=str, help="Server to ping")
    parser.add_argument("port", type=int, help="Port to ping")
    parser.add_argument("--timeout", "-t", type=int, help="Timeout in seconds", default=3)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    ping_result = ping_server(args.server, args.port, args.timeout)
    if args.verbose:
        print(ping_result)
    if ping_result:
        exit(0)
    exit(1)
