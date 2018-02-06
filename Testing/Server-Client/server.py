from socket import *

s = socket(AF_INET, SOCK_STREAM)
s.bind(('192.168.0.60', 25004))
s.listen(0)
print("[*] Waiting for a connection...")
while True:
    conn, addr = s.accept()
    print("[*] Successfully Connected to", addr)
    while True:
        f = open('forward.png', 'wb')
        print("[*] Recieving the file...")
        l = conn.recv(15000)
        while (l):
            f.write(l)
            l=conn.recv(15000)
            print("Got it")
        print("[*] Out of loop - file recieved")
        break
    break

conn2, addr2 = s.accept()
print("[*] Second Connection Accepted")
data = ""
conn2.sendall(data)
