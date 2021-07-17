import argparse
from UDP_methods import UDP_main

# parsing user input option
parser = argparse.ArgumentParser(description='Train Implementation')
parser.add_argument('--host_ip', type = str, default ="127.0.0.1", help='host_ip')
parser.add_argument('--client_ip', type = str, default ="127.0.0.1", help='client_ip')
parser.add_argument('--host_port', type = int, default =8888, help='host_port')
parser.add_argument('--client_port', type = int, default =8889, help='client_port')
parser.add_argument('--yolo_port', type = int, default =8886, help='yolo_port')
args = parser.parse_args()


# main loop
if __name__ == "__main__":
    host_ip = args.host_ip
    client_ip = args.client_ip
    host_port = args.host_port
    client_port = args.client_port
    yolo_port = args.yolo_port
    UDP_main(host_ip, host_port, client_ip, client_port, yolo_port)
