from socket import *
import numpy

def send_from(arr, dest):
    view = memoryview(arr).cast('B')
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]

s = socket(AF_INET, SOCK_STEAM)
s.bind(('', 25000))
s.listen(1)
c, a = s.accept()

a = numpy.arange(0.0, 50000000.0)
send_from(a, c)
