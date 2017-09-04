# CHANGE CODE AS TAKEN FROM STACKOVERFLOW [PLAGARISM]
import socket               

s = socket.socket()         
host = 'localhost' 
port = 12346                 
s.bind(('localhost', port))        
f = open('torecv.png','wb')
s.listen(1)                 
try:
    while True:
        c, addr = s.accept()     
        print ('Got connection from', addr)
        print ("Receiving...")
        l = c.recv(90024)
        while (l):
            print ("Receiving...")
            f.write(l)
            l = c.recv(90024)
        f.close()
        print ("Done Receiving")
        c.send(b'Thank you for connecting')
        c.close()
        break
except KeyboardInterrupt:
    c.close()
    print ("Keyboard Interrupt")
