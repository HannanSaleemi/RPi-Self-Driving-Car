#define ENABLE 5      //pwm
#define DIRA 4        //1/0
#define DIRB 3
#define ENABLE2 6   //pwm 2
#define DIRC 7
#define DIRD 8

char receivedChar;
boolean newData = false;

void setup() {
  // put your setup code here, to run once:
  pinMode(ENABLE,OUTPUT);
  pinMode(DIRA,OUTPUT);
  pinMode(DIRB,OUTPUT);
  pinMode(ENABLE2,OUTPUT);
  pinMode(DIRC,OUTPUT);
  pinMode(DIRD,OUTPUT);
  Serial.begin(9600);
  delay(2000);
  
}

void loop() {

  analogWrite(ENABLE2, 1023);
  digitalWrite(DIRC,HIGH);
  digitalWrite(DIRD,LOW);
  delay(1000);
  digitalWrite(DIRD,HIGH);
  digitalWrite(DIRC,LOW);
  delay(1000);
  analogWrite(ENABLE, 225);
  digitalWrite(DIRA,HIGH);
  digitalWrite(DIRB,LOW);
  delay(500);

  recvInfo();

  /*
  if newData == "Left" {
    left();
    newData = false;
  }
  if newData == "Right"{
    right();
    newData = false;
  }
  if newData = "Forward"{
    forward();
    newData = false;
  }
  */
  
  //left();
  //delay(2000);
  //right();
  //delay(2000);
  //forward();
  //delay(2000);

  //LEFT AND RIGHT TURN MOTORS CONFIRM WORKING
  
  /*
   /working code below
  analogWrite(ENABLE2, 1023);
  digitalWrite(DIRC,HIGH);
  digitalWrite(DIRD,LOW);
  delay(1000);
  digitalWrite(DIRC,LOW);
  digitalWrite(DIRD,HIGH);
  delay(1000);

  analogWrite(ENABLE, 225);
  digitalWrite(DIRA,HIGH);
  digitalWrite(DIRB,LOW);
  delay(1000);
  digitalWrite(DIRA,LOW);
  digitalWrite(DIRB,HIGH);
  delay(1000);

  */
  

  
  //128 +  SPEED PWM WORKING FOWARD MOTORS
  
  
  // put your main code here, to run repeatedly:
  /*
  analogWrite(ENABLE, 255);
  digitalWrite(DIRA,HIGH); //one way
  digitalWrite(DIRB,LOW);
  delay(5000);
  Serial.print("90 BEGINS");
  analogWrite(ENABLE,90); //half speed
  delay(5000);
  Serial.print("90 STOP");
  digitalWrite(ENABLE,LOW); //all done
  delay(4000);
  */

  //analogWrite(ENABLE2, 255);
  //digitalWrite(DIRC,HIGH);
  //digitalWrite(DIRD,LOW);
  //analogWrite(ENABLE,255);
  //digitalWrite(DIRA,HIGH);
  //digitalWrite(DIRB,LOW);  
 // delay(5000);

  //digitalWrite(DIRA,LOW);
  //digitalWrite(DIRB,LOW);
  
  //digitalWrite(DIRC,LOW);
  //digitalWrite(DIRD,HIGH);
  //digitalWrite(DIRA,LOW);
  //digitalWrite(DIRB,HIGH);
  //delay(5000);
}

void recvInfo() {

  if (Serial.available() > 0) {

    receivedChar = Serial.read();
    newData = true;
    
}
}

void left(){
  analogWrite(ENABLE2, 1023);
  digitalWrite(DIRC,LOW);
  digitalWrite(DIRD,HIGH);
  analogWrite(ENABLE, 200);
  digitalWrite(DIRA,HIGH);
  digitalWrite(DIRB,LOW);
}

void right(){
  analogWrite(ENABLE2, 1023);
  digitalWrite(DIRC,HIGH);
  digitalWrite(DIRD,LOW);
  analogWrite(ENABLE, 200);
  digitalWrite(DIRA,HIGH);
  digitalWrite(DIRB,LOW);
}


void forward(){
  analogWrite(ENABLE2, 0);
  analogWrite(ENABLE, 200);
  digitalWrite(DIRA,HIGH);
  digitalWrite(DIRB,LOW);
}

