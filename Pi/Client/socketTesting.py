from socket import *
import numpy

def recv_into(arr, source):
    view = memoryview(arr).cast('B')
    while len(view):
        nrecv = source.recv_into(view)
        view = view[nrecv:]

c = socket(AF_INET, SOCK_STREAM)
c.connect(('192.168.0.10', 25000))

a = numpy.zeros(shape=50000000, dtype=float)
print(a[0:10
        ])
recv_into(a, c)

print(a[0:10])
