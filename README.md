# RPi-Self-Driving-Car

**This Project was for my Computer Science Coursework~**
**It is no longer being maintained. I will create a better version in the future.**

Raspberry pi based RC Self-Driving Car using Python, Tenserflow and OpenCV

## **Iteration 1**
  - [x] Data Pre-Processing script to process the dataset `DataPreProcessing.py`
  - [x] Training script to train the model `training.py`
  - [x] Model:
    - Training Accuracy - 99.9%
    - Testing Accuracy - 89.7%
    - Model Saved to disk

## **Iteration 2**
  - [X] Client 1 (Pi)
    - Send Snapshot
    - Recieve result
    - Send result to Arduino
  - [X] Server
    - Receive Image
    - Generate Prediction
      - Directional Prediction
      - Traffic Light Detection
      - STOP sign detection
    - Send results to Arduino (Client 2)

## **Iteration 3**
  - [X] Client 1
    - Send result to Client 2
  - [X] Client 2
    - Recieve result
    - Parse:
      - Get distance from ultrasonic sensor
      - Execute correct driving function


Mac (Server) -> Pi (Client 1) -> Arduino (Client 2)
