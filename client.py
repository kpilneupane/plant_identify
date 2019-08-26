import socket
import cv2
HOST = '127.0.10.1'
PORT = 65432
image_size=32
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    #here needs to check the image type
    inputimage=cv2.imread('/home/kafle/Pictures/test/Acer_Campestre_15.ab.jpg',0)
    laplacian = cv2.Laplacian(inputimage,cv2.CV_64F)
    my_image=cv2.resize(laplacian,(image_size,image_size))
    bytearray_image=bytearray(my_image)
    s.send(inputimage)
    fetchmedicalvalue=s.recv(1024)
    fetchtopologicalRegion=s.recv(1024)
    fetchhumidity=s.recv(1024)
    fetchusefulPart=s.recv(1024)
    fetchimage=s.recv(1024)
    s.close()
    print(fetchmedicalvalue)
    print(fetchtopologicalRegion)
    print(fetchhumidity)
    print(fetchusefulPart)
    print(fetchimage)
