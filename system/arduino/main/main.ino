#include <IRremote.h>
#include <Servo.h>

#define PIN_DETECT 2

IRsend irsend;
Servo servo;  // Create object for Servo motor

int position = 80;
String message;
bool detected = false;

void setup() {
  // put your setup code here, to run once:
  pinMode(PIN_DETECT, INPUT); //initialize IR receiver as input.
  pinMode(LED_BUILTIN, OUTPUT);// initialize digital pin LED_BUILTIN as an output.

  irsend.enableIROut(38);
  irsend.mark(0);

  servo.attach(3, 500, 2430);   // Set PWM pin 3 for Servo motor 1
  servo.write(position);

  Serial.begin(9600);// initialize serial communication @ 9600 baud:
  Serial.println("Arduino started!");
}

void open_gate(){
  // Rotating Servo motor in clockwise from 0 degree to 90 degree
  for (position = 80; position < 220; position+=10) 
  { 
    servo.write(position);  // Set position of Servo motor 
    delay(10);               // Short delay to control the speed      
  }
  delay(3000);
}

void close_gate(){
  // Rotating Servo motor in clockwise from 0 degree to 90 degree
  for (position = 220; position > 80; position-=10) 
  { 
    servo.write(position);  // Set position of Servo motor 
    delay(10);               // Short delay to control the speed      
  }
}

void loop() {
  // put your main code here, to run repeatedly:
  if (digitalRead(PIN_DETECT) == 1 )
    {
      if (detected == false) {
        digitalWrite(LED_BUILTIN, HIGH); // Led ON
        Serial.println("[arduino-sent] Object detected!");
        detected = true;
      }

      // if(position!=220)
      //   open_gate();
      if(Serial.available()){
      message = String(Serial.readString());
      message.trim();
      if(message.substring(0) == "[python-sent] Access allowed."){
        if(position!=220)
          open_gate();
      }
    }
    }
    else
    {
      digitalWrite(LED_BUILTIN, LOW); // Led OFF
      detected = false;
      Serial.println("[arduino-sent] No Object Detected");
      if(position!=80)
        close_gate();
    }
    delay(1000);
}
