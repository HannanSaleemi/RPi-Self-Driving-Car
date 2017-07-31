# CHANGE FILE AS IT IS TAKEN FROM STACKOVERFLOW [PLAGARISM]
import socket              

s = socket.socket()         
               
s.connect(('localhost', 12346))
f = open('/Users/intern.mac/Desktop/RPi-Self-Driving-Car/Client-Server Test/tosend.png','rb')
print('Sending...')
l = f.read(90024)
while (l):
    print('Sending...')
    s.send(l)
    l = f.read(90024)
f.close()
print("Done Sending")
s.shutdown(socket.SHUT_WR)
print(s.recv(90024))
s.close()
