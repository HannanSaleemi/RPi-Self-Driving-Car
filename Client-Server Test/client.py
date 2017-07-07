# CHANGE FILE AS IT IS TAKEN FROM STACKOVERFLOW [PLAGARISM]
import socket              

s = socket.socket()         
               
s.connect(('192.168.0.19', '8081'))
f = open('tosend.png','rb')
print('Sending...')
l = f.read(1024)
while (l):
    print('Sending...')
    s.send(l)
    l = f.read(1024)
f.close()
print("Done Sending")
s.shutdown(socket.SHUT_WR)
print(s.recv(1024))
s.close()
