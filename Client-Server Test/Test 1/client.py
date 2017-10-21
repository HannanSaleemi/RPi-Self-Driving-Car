import socket              

s = socket.socket()         
               
s.connect(('localhost', 12347))
f = open('tosend.png','rb')
print('Sending...')
l = f.read(150000)
while (l):
    print('Sending...')
    s.send(l)
    l = f.read(150000)
f.close()
print("Done Sending")
s.shutdown(socket.SHUT_WR)
print(s.recv(150000))
s.close()
