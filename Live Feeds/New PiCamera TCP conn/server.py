import socket

HOST = '192.168.0.60'
PORT = 1249
ADDR = (HOST,PORT)
BUFSIZE = 4096

serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

serv.bind(ADDR)
serv.listen(5)

print "Listening for a connection from the server..."

while True:
    conn, addr = serv.accept()
    print "Connected to client ", addr

    myfile = open('img.png', 'wb')

    while True:
        data = conn.recv(BUFSIZE)
        if not data: break
        myfile.write(data)
        print "Writing File..."

    myfile.close()
    print "Finished"
    conn.close()
