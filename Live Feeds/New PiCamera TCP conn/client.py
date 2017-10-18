from socket import *
import picamera
from PIL import Image
import StringIO

HOST = '192.168.0.60'
PORT = 1247
ADDR = (HOST,PORT)
BUFSIZE = 4096

camera = picamera.PiCamera()
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

camera.capture('img.png')

buf = StringIO.StringIO()
img = open('img.png', 'rb')
img.save(buf,format='PNG')
client.sendall(buf.getvalue())

client.close()

    
