#Will need adapting to avaoid plagerism
#https://gist.github.com/marvin/4318413
import socket

HOST = 'localhost'
PORT = '9876'
ADDR = (HOST, PORT)
BUFSIZE = 4096

videofile = "videos/file.mp4"

bytes = open(videofile).read()

print len(bytes)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

client.send(bytes)

client.close()
