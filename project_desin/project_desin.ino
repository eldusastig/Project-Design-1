#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <ArduinoMqttClient.h>
#include <TinyGPSPlus.h>

// ─── Pin definitions ───────────────────────────────────────────────────────────────
#define GPS_RX           16
#define GPS_TX           17
#define WATER_SENSOR_PIN 32
#define ULTRASONIC_TRIG  25
#define ULTRASONIC_ECHO  26

// ─── Device ID ─────────────────────────────────────────────────────────────────────
const char* deviceId = "DVC001";  // unique per device

// ─── Wi-Fi Credentials ─────────────────────────────────────────────────────────────
const char* ssid      = "GlobeAtHome_34434_2.4";
const char* wifi_pass = "5crNrNuk";

// ─── HiveMQ Cloud (MQTT over TLS) ───────────────────────────────────────────────────
const char* mqttBroker   = "a62b022814fc473682be5d58d05e5f97.s1.eu.hivemq.cloud";
const int   mqttPort     = 8883;  // TLS port
const char* mqttUser     = "prototype";
const char* mqttPassword = "Prototype1";

// ─── MQTT Topics ────────────────────────────────────────────────────────────────────
const char* gpsTopic     = "esp32/gps";
const char* floodTopic   = "esp32/sensor/flood";
const char* binFullTopic = "esp32/sensor/bin_full";

bool collecting = false;
// ─── Globals ────────────────────────────────────────────────────────────────────────
WiFiClientSecure net;
MqttClient       client(net);
TinyGPSPlus      gps;
HardwareSerial   gpsSerial(2);

unsigned long lastPubGPS     = 0;
unsigned long lastPubSensors = 0;
const unsigned long GPS_INTERVAL     = 5000;
const unsigned long SENSORS_INTERVAL = 2000;

// ─── Ultrasonic helper ──────────────────────────────────────────────────────────────
long readUltrasonicCM() {
  digitalWrite(ULTRASONIC_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIG, LOW);

  long duration = pulseIn(ULTRASONIC_ECHO, HIGH, 30000UL);
  if (duration == 0) return -1;
  return duration / 29 / 2;
}

// ─── Wi-Fi Connect ─────────────────────────────────────────────────────────────────
void connectWiFi() {
  Serial.print("Connecting to Wi-Fi");
  WiFi.begin(ssid, wifi_pass);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi connected! IP: " + WiFi.localIP().toString());
}

// ─── MQTT Connect ─────────────────────────────────────────────────────────────────
void connectMQTT() {
  String clientId = String("esp32-") + deviceId + "-" + String(random(0xffff), HEX);
  client.setId(clientId.c_str());
  client.setUsernamePassword(mqttUser, mqttPassword);

  Serial.print("Connecting to MQTT broker");
  while (!client.connect(mqttBroker, mqttPort)) {
    Serial.print(".");
    delay(1000);
  }
  Serial.println("\nMQTT connected!");
}

void setup() {
  Serial.begin(115200);
  delay(100);

  pinMode(WATER_SENSOR_PIN, INPUT_PULLUP);
  pinMode(ULTRASONIC_TRIG, OUTPUT);
  pinMode(ULTRASONIC_ECHO, INPUT);

  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);

  connectWiFi();
  net.setInsecure();   // for testing only
  connectMQTT();
}

void loop() {
  client.poll();
  unsigned long now = millis();
  if (Serial.available()):
   String cmd=Serial.ReadStringUntil('\n');
   cmd.trim();
  if(cmd=="Collect"){
    collecting=true;
    digitalWrite(CollectionPin,HIGH);
    Serial.println("Collection Started");
  }
  

    
  // ─── 1) GPS ──────────────────────────────────────────────────────────────────────
  while (gpsSerial.available()) {
    gps.encode(gpsSerial.read());
  }
  if (now - lastPubGPS >= GPS_INTERVAL && gps.location.isUpdated()) {
    float lat = gps.location.lat();
    float lon = gps.location.lng();
    String payload = String("{\"id\":\"") + deviceId +
                     String("\",\"lat\":")  + String(lat, 6) +
                     String(",\"lon\":")   + String(lon, 6) +
                     String("}");
    client.beginMessage(gpsTopic);
    client.print(payload);
    client.endMessage();
    Serial.println("Published GPS: " + payload);
    lastPubGPS = now;
  }

  // ─── 2) Sensors ──────────────────────────────────────────────────────────────────
  if (now - lastPubSensors >= SENSORS_INTERVAL) {
    // Flood sensor
    bool flooded = (digitalRead(WATER_SENSOR_PIN) == LOW);
    String floodPayload = String("{\"id\":\"") + deviceId +
                          String("\",\"flooded\":") + (flooded ? "true" : "false") +
                          String("}");
    client.beginMessage(floodTopic);
    client.print(floodPayload);
    client.endMessage();
    Serial.println("Published Flood: " + floodPayload);

    // Bin‐full (ultrasonic)
    long distanceCM = readUltrasonicCM();
    bool binFull = (distanceCM > 0 && distanceCM < 10);
    String binPayload = String("{\"id\":\"") + deviceId +
                        String("\",\"binFull\":") + (binFull ? "true" : "false") +
                        String("}");
    client.beginMessage(binFullTopic);
    client.print(binPayload);
    client.endMessage();
    Serial.println("Published Bin Full: " + binPayload);

    lastPubSensors = now;
  }

  delay(10);
}
