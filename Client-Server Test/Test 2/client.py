import socket
import picamera

camera = picamera.PiCamera()

camera.capture('tosend.png')

s = socket.socket()         
               
s.connect(('192.168.60', 12347))
f = open('tosend.png','rb')
print('Sending...')
l = f.read(200000)
while (l):
    print('Sending...')
    s.send(l)
    l = f.read(200000)
f.close()
print("Done Sending")
s.shutdown(socket.SHUT_WR)
print(s.recv(200000))
s.close()
