//Ultrasonic Sensor Setup
#include <SR04.h>
#define TRIG_PIN 12
#define ECHO_PIN 13

//Forward Motors
#define ENABLE 5 //pwm
#define DIRA 3 //1/0
#define DIRB 4 //0/1

//Turn Motors
#define ENABLETURN 10 //pwm turn
#define DIRC 9 //1/0
#define DIRD 8 //0/1

//Ultrasonic sensor declarations
long distance;
SR04 sr04 = SR04(ECHO_PIN, TRIG_PIN);

//Recieving Bytes
int incomingByteFirst;
int incomingByteSecond;
int incomingByteThird;

void setup() {
  //Runs once
  Serial.begin(115200);
  pinMode(ENABLE, OUTPUT);
  pinMode(DIRA, OUTPUT);
  pinMode(DIRB, OUTPUT);
  pinMode(ENABLETURN, OUTPUT);
  pinMode(DIRC, OUTPUT);
  pinMode(DIRD, OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

void loop() {
  
  if (Serial.available() > 0){
    //Read the bytes
    incomingByteFirst = Serial.read();
    delay(50);
    incomingByteSecond = Serial.read();
    delay(50);
    incomingByteThird = Serial.read();

    //Print recieved bytes
    Serial.println(incomingByteFirst);
    Serial.println(incomingByteSecond);
    Serial.println(incomingByteThird);

    if (incomingByteFirst == -1 or incomingByteSecond == -1 or incomingByteThird == -1){
      //Invalid Command - Three Commands not recieved
      Serial.println("Invalid Command");
    }

    //Get Distance
    distance = sr04.Distance();

    if (distance <= 15){
      //Do Nothing if distance is 20cm or less
      Serial.println(distance);
      Serial.println("Stopping...");
    }
    else if (incomingByteThird == 84){
      //STOP SIGN DETECTED DO NOTHING
      Serial.println("STOP DETECTED - DO NOTHING");
    }
    else if (incomingByteSecond == 82) {
      //RED LIGHT DETECTED STAY STILL
      Serial.println("RED LIGHT DETECTED - STOPPING VEHICLE");
    }
    else if (incomingByteFirst == 48){
      //If the first byte = 0 (ASCII DEC: 48) - Drive forward
      Serial.println("FORWARD");
      analogForward();
    }
    else if (incomingByteFirst == 49){
      //If the first byte = 1 (ASCII DEC: 49) - Drive Left
      Serial.println("LEFT");
      analogLeft();
    }
    else if (incomingByteFirst == 50){
      //If the first byte = 2 (ASCII DEC: 50) - Drive Right
      Serial.println("RIGHT");
      analogRight();
    }
  }
}
/*  
  if (Serial.available() > 0){
    //Read the bytes
    incomingByteFirst = Serial.read();
    delay(50);
    incomingByteSecond = Serial.read();
    delay(50);
    incomingByteThird = Serial.read();

    //Print recieved bytes
    Serial.println(incomingByteFirst);
    Serial.println(incomingByteSecond);
    Serial.println(incomingByteThird);

    //Invalid Command - Three Commands not recieved
    if (incomingByteFirst == -1 or incomingByteSecond == -1 or incomingByteThird == -1){
      Serial.println("Invalid Command");
    }
    
    if (incomingByteFirst == 48){
      //If the first byte = 0 (ASCII DEC: 48) - Drive forward
      Serial.println("FORWARD");
      analogForward();
    }

    if (incomingByteFirst == 50){
      //If the first byte = 2 (ASCII DEC: 50) - Drive Right
      Serial.println("RIGHT");
      analogRight();
    }

    if (incomingByteFirst == 49){
      //If the first byte = 1 (ASCII DEC: 49) - Drive Left
      Serial.println("LEFT");
      analogLeft();
    }
  }
}
*/
void analogForward() {
  //Enable for 90ms - Then stop
  digitalWrite(ENABLETURN, LOW);
  digitalWrite(DIRC, LOW);
  digitalWrite(DIRD, LOW);  
  digitalWrite(ENABLE, HIGH);
  digitalWrite(DIRA, HIGH);
  digitalWrite(DIRB, LOW);
  delay(90);
  digitalWrite(ENABLE, LOW);
  digitalWrite(DIRA, LOW);
  digitalWrite(DIRB, LOW);
}

void turnForward() {
  //Enable for 90ms - Then stop
  digitalWrite(ENABLE, HIGH);
  digitalWrite(DIRA, HIGH);
  digitalWrite(DIRB, LOW);
  delay(90);
  digitalWrite(ENABLE, LOW);
  digitalWrite(DIRA, LOW);
  digitalWrite(DIRB, LOW);  
}

void analogRight() {
  //Drive Left - approx time 90ms
  digitalWrite(ENABLETURN, HIGH);
  digitalWrite(DIRC, LOW);
  digitalWrite(DIRD, HIGH);
  delay(90);
  turnForward();
}

void analogLeft() {
  //Drive Right - approx time 50ms
  digitalWrite(ENABLETURN, HIGH);
  digitalWrite(DIRC, HIGH);
  digitalWrite(DIRD, LOW);
  delay(90);
  turnForward();
}

