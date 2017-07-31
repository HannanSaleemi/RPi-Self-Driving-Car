/* MOTOR 1 */
#define ENABLE 5
#define DIRA 4
#define DIRB 3

/* MOTOR 2 */
#define ENABLE 6
#define DIRC 7
#define DIRD 8

/* VARIABLE DECLERATIONS */
char recivedData;							/* Recieved Data from the Raspberry Pi VIA USB */
boolean newData = false;

/* SET PINMODES AND SERIAL RATE */
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
	recvInfo();

	if receivedChar == "L"{
		left();
		newData = false;
	}
	if receivedChar == "R"{
		right();
		newData = false;
	}
	if receivedChar == "F"{
		forward();
		newData = false;
	}

}

void recvInfo() {
	if (Serial.available() > 0) {
    	receivedChar = Serial.read();
    	newData = true;
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
