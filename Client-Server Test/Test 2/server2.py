import socket               

s = socket.socket()         
host = '192.168.0.10' 
port = 12346                 
s.bind(('192.168.0.10', port))        
f = open('pirecv.png','wb')
s.listen(1)
while True:
    try:
        c, addr = s.accept()
        print ("Got Connection from", addr)
        l = c.recv(90024)
        while (l):
            print ("Recieving...")
            f.write(l)
            l = c.recv(90024)
        f.close()
        print ("Done Reciving...")
    except Exception:
        c.close()
        print ("Done!")
