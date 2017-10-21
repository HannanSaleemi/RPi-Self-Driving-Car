import socket               

s = socket.socket()         
host = 'localhost' 
port = 12347                 
s.bind(('localhost', port))        
f = open('torecv.png','wb')
s.listen(1)                 
try:
    while True:
        c, addr = s.accept()     
        print ('Got connection from', addr)
        print ("Receiving...")
        l = c.recv(150000)
        while (l):
            print ("Receiving...")
            f.write(l)
            l = c.recv(150000)
        f.close()
        print ("Done Receiving")
        c.send(b'Thank you for connecting')
        c.close()
        break

    #Neural Network Predition START
    from keras.models import load_model
    classifier = load_model("/Users/intern.mac/Desktop/RPi-Self-Driving-Car/Trained Models/28-july-car_cnn.h5")

    import numpy as np
    from keras.preprocessing import image

    test_image = image.load_img('torecv.png',  target_size=(64, 64))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis=0)

    result = classifier.predict(test_image)

    print result
    #Neural Network Prediction END
    
except KeyboardInterrupt:
    c.close()
    print ("Keyboard Interrupt")

