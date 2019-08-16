import socket
import cv2
HOST = '127.0.0.1'
PORT = 65432
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    #here needs to check the image type
    inputimage=cv2.imread()
    bytearray_image=bytearray(inputimage)
    s.sendall(bytearray)
    s.close()
