# CHANGE CODE AS TAKEN FROM STACKOVERFLOW [PLAGARISM]
import socket               

s = socket.socket()         
host = socket.gethostname() 
port = 12345                 
s.bind((host, port))        
f = open('torecv.png','wb')
s.listen(1)                 
while True:
    c, addr = s.accept()     
    print ('Got connection from', addr)
    print ("Receiving...")
    l = c.recv(1024)
    while (l):
        print ("Receiving...")
        f.write(l)
        l = c.recv(1024)
    f.close()
    print ("Done Receiving")
    c.send(b'Thank you for connecting')
    c.close()                
