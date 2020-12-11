

#define echoPin 2 // attach pin D2 Arduino to pin Echo of HC-SR04
#define trigPin 3 //attach pin D3 Arduino to pin Trig of HC-SR04
const int buzzer = 9;
#define ledpin1 4
#define ledpin2 5
#define ledpin3 6
// defines variables
long duration; // variable for the duration of sound wave travel
int distance; // variable for the distance measurement

void setup() {
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an OUTPUT
  pinMode(echoPin, INPUT); // Sets the echoPin as an INPUT
  Serial.begin(9600); // // Serial Communication is starting with 9600 of baudrate speed
  pinMode(ledpin1,OUTPUT);
  pinMode(ledpin2,OUTPUT);
  pinMode(ledpin3,OUTPUT);
  pinMode(buzzer, OUTPUT);
}
void loop() {
  // Clears the trigPin condition
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  // Sets the trigPin HIGH (ACTIVE) for 10 microseconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);
  // Calculating the distance
  distance = duration * 0.034 / 2; // Speed of sound wave divided by 2 (go and back)
  // Displays the distance on the Serial Monitor
  Serial.print(distance);
  Serial.print("\n");
  delay(100);
  if(distance<150)
  {
    digitalWrite(ledpin1,HIGH);
    delay(100);
   
  }
  if (distance<100)
  {
    digitalWrite(ledpin1,HIGH);
    delay(100);
   
    digitalWrite(ledpin2,HIGH);
    delay(100);
    
  }
  if(distance<50)
  {
    tone(buzzer, 1000); // Send 1KHz sound signal...
    digitalWrite(ledpin1,HIGH);
    delay(100);
    digitalWrite(ledpin2,HIGH);
    delay(100);
    digitalWrite(ledpin3,HIGH);
    delay(100);
    digitalWrite(ledpin1,LOW);
    delay(100);
    digitalWrite(ledpin2,LOW);
    delay(100);
    digitalWrite(ledpin3,LOW);
    delay(100);
 
  }
  else if (distance>150)
  {
    noTone(buzzer); 
    digitalWrite(ledpin1,LOW);
    delay(100);
    digitalWrite(ledpin2,LOW);
    delay(100);
    digitalWrite(ledpin3,LOW);
    delay(100);
    
  }
}
