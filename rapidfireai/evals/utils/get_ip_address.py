"""
Get the IP address of the machine.
"""

def get_ip_address():
    """ Get the IP address of the machine. """
    import socket
    try:
        return socket.gethostbyname(socket.gethostname())
    except socket.gaierror as e:
        return "<server_ip_or_hostname>"