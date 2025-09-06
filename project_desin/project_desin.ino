// Full ESP32 sketch (with ADC water sensor + robust HC-SR04 reads + collect queue/watchdog)
// IMPORTANT: ensure HC-SR04 echo pin is level-shifted to 3.3V before connecting to ESP32.

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <ArduinoMqttClient.h>
#include <TinyGPSPlus.h>
#include <vector>
#include <algorithm>
#include <math.h> // for roundf, fabs

// ─── Pin definitions ───────────────────────────────────────────────────────────────
#define GPS_RX            16
#define GPS_TX            17
#define WATER_SENSOR_PIN  34   // ADC-capable pin (34..39 on many ESP32 boards)

#define ULTRASONIC_TRIG1   33 
#define ULTRASONIC_ECHO1   32
#define ULTRASONIC_TRIG2   27
#define ULTRASONIC_ECHO2   26
#define ULTRASONIC_TRIG3   25
#define ULTRASONIC_ECHO3   23
#define ULTRASONIC_TRIG4    4
#define ULTRASONIC_ECHO4   35

#define RPWM               22
#define LPWM               21
#define REN                19
#define LEN                18

// Buzzer pin (added)
#define BUZZER_PIN         14   // change to any free GPIO you prefer

// If your sensor is inverted (HIGH means water present), set to 1
#define WATER_SENSOR_INVERTED 1

// Motor wiring mode (uncomment for single PWM)
// #define SINGLE_PWM 1
#ifdef SINGLE_PWM
  #define PWM_PIN RPWM
  #define DIR_PIN REN
#endif

// ─── Device ID / Wi-Fi / MQTT ──────────────────────────────────────────────────────
const char* deviceId = "DVC006";
const char* ssid      = "meltryllis ikuwayo"; 
const char* wifi_pass = "ikuwayo ikuwayo";

const char* mqttBroker   = "a62b022814fc473682be5d58d05e5f97.s1.eu.hivemq.cloud";
const int   mqttPort     = 8883;
const char* mqttUser     = "prototype";
const char* mqttPassword = "Prototype1";

const char* gpsTopic     = "esp32/gps";
const char* floodTopic   = "esp32/sensor/flood";
const char* binFullTopic = "esp32/sensor/bin_full";
String detectionTopic() { return String("esp32/") + deviceId + "/sensor/detections"; }

// ─── Globals ───────────────────────────────────────────────────────────────────────
WiFiClientSecure net;
MqttClient       client(net);
TinyGPSPlus      gps;
HardwareSerial   gpsSerial(2);

unsigned long lastPubGPS     = 0;
unsigned long lastPubSensors = 0;
const unsigned long GPS_INTERVAL     = 5000;
const unsigned long SENSORS_INTERVAL = 2000;

// Motor timings (up and down equal)
enum MotorState { IDLE, GOING_UP, WAIT_AFTER_UP, GOING_DOWN };
MotorState state = IDLE;
unsigned long motorStart = 0;
const unsigned long motorUP   = 3260; // ms to go UP (3 seconds)
const unsigned long motorDOWN = 2900; // ms to go DOWN (3 seconds)
const unsigned long idleAfterUpMs = 1000; // 1s idle at top

// detection publish debounce
unsigned long lastDetectionPublishMs = 0;
String lastDetectionPayload = "";
const unsigned long DETECTION_MIN_INTERVAL_MS = 800; // tune for your use

// NEW: queued detection payload to publish with periodic sensors
bool pendingDetection = false;
String pendingDetectionPayload = "";

// Flood debounce globals
const unsigned long FLOOD_DEBOUNCE_MS = 800; // ms
bool floodStableState = false;
unsigned long floodLastChangeMs = 0;

// ADC water sensor tuning
const int WATER_THRESHOLD = 1500;
const adc_attenuation_t WATER_ADC_ATTEN = ADC_11db; // Arduino-ESP32 API

// HC-SR04 tuning
const unsigned long HCSR04_TIMEOUT_US = 30000UL; // 30 ms timeout (~5 m)
const int HCSR04_ATTEMPTS = 3;                   // number of pings to average
const int BIN_FULL_CM = 10;                      // threshold for "bin full" (tweak as needed)

// Ultrasonic mount height: sensor mounted 12 inches above bin bottom
const float ULTRASONIC_MOUNT_IN = 12.0f;
const float ULTRASONIC_MOUNT_CM = ULTRASONIC_MOUNT_IN * 2.54f; // ~30.48 cm

// --- Ultrasonic helper: multiple pings + averaging (returns -1 if no valid echo) ---
long readUltrasonicCM(uint8_t trigPin, uint8_t echoPin, int attempts = HCSR04_ATTEMPTS) {
  long sum = 0;
  int valid = 0;
  for (int i = 0; i < attempts; ++i) {
    // ensure trig is LOW
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);

    // 10us trigger pulse
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    // wait for echo (timeout prevents blocking forever)
    unsigned long duration = pulseIn(echoPin, HIGH, HCSR04_TIMEOUT_US);

    if (duration > 0) {
      long cm = (long)(duration / 29UL / 2UL); // approximate conversion
      sum += cm;
      valid++;
    } else {
    }

    // short delay between pings
    delay(10);
  }

  if (valid == 0) return -1;
  return sum / valid; // integer average
}

// --- Motor helpers (unchanged) ---
void motor_start_up() {
#ifdef SINGLE_PWM
  digitalWrite(DIR_PIN, HIGH); // HIGH = UP
  analogWrite(PWM_PIN, 255);
#else
  analogWrite(RPWM, 255);
  analogWrite(LPWM, 0);
#endif
}
void motor_start_down() {
#ifdef SINGLE_PWM
  digitalWrite(DIR_PIN, LOW); // LOW = DOWN
  analogWrite(PWM_PIN, 255);
#else
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 255);
#endif
}
void motor_stop() {
#ifdef SINGLE_PWM
  analogWrite(PWM_PIN, 0);
#else
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);
#endif
}

// --- WiFi / MQTT helpers (unchanged) ---
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.begin(ssid, wifi_pass);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 10000) {
    delay(500);
    Serial.print('.');
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWi-Fi connected: " + WiFi.localIP().toString());
  } else {
    Serial.println("\nWi-Fi failed, continuing offline.");
  }
}
void connectMQTT() {
  if (WiFi.status() != WL_CONNECTED) return;
  if (client.connected()) return;
  String clientId = String("esp32-") + deviceId + "-" + String(random(0xffff), HEX);
  client.setId(clientId.c_str());
  client.setUsernamePassword(mqttUser, mqttPassword);
  String statusTopic = String("esp32/") + deviceId + "/status";
  String willPayload  = String("{\"id\":\"") + deviceId + "\",\"status\":\"offline\"}";
  client.beginWill(statusTopic.c_str(), willPayload.length(), true, 1);
  client.print(willPayload);
  client.endWill();
  Serial.print("Connecting MQTT...");
  if (client.connect(mqttBroker, mqttPort)) {
    Serial.println("MQTT connected!");
    String onlinePayload = String("{\"id\":\"") + deviceId + String("\",\"status\":\"online\"}");
    client.beginMessage(statusTopic.c_str(), onlinePayload.length(), true, 1);
    client.print(onlinePayload);
    client.endMessage();
  } else {
    Serial.println("MQTT connect failed, offline mode.");
  }
}

// --- Detection parsing / publishing helpers (unchanged) ---
String strTrim(const String &s) { int a=0,b=s.length()-1; while(a<=b && isspace(s[a])) a++; while(b>=a && isspace(s[b])) b--; if (a==0 && b==s.length()-1) return s; return s.substring(a,b+1); }
std::vector<String> splitClasses(const String &s) { std::vector<String> out; int i=0; while(i<s.length()){ int j=s.indexOf(',',i); String part; if(j==-1){ part=s.substring(i); i=s.length(); } else { part=s.substring(i,j); i=j+1; } part.trim(); if(part.startsWith("\"")&&part.endsWith("\"")&&part.length()>=2) part=part.substring(1,part.length()-1); part.trim(); if(part.length()) out.push_back(part);} return out; }
std::vector<String> parseClassesFromJson(const String &line){ std::vector<String> out; String low=line; low.toLowerCase(); int idx=low.indexOf("classes"); if(idx==-1) return out; int b=line.indexOf('[', idx); int e=(b==-1)?-1:line.indexOf(']', b); if(b==-1||e==-1||e<=b) return out; String inside=line.substring(b+1,e); auto parts = splitClasses(inside); for(auto &p:parts) out.push_back(p); return out;}
std::vector<String> parseClasses(const String &raw){ std::vector<String> out; String s=strTrim(raw); if(s.length()==0) return out; String up=s; up.toUpperCase(); if(up.startsWith("DETECTIONS:")||up.startsWith("CLASSES:")||up.startsWith("CLASS:")){ int colon=s.indexOf(':'); if(colon!=-1){ String body=s.substring(colon+1); return splitClasses(body);} } auto j=parseClassesFromJson(s); if(!j.empty()) return j; if(s.indexOf(' ')==-1 && s.indexOf('{')==-1 && s.indexOf('[')==-1 && s.indexOf(':')==-1) out.push_back(s); return out;}
void buildOrderedCounts(const std::vector<String> &classes, std::vector<String> &ordered, std::vector<int> &counts, std::vector<String> &keys){ ordered.clear(); counts.clear(); keys.clear(); for(auto &c:classes){ bool found=false; for(auto &u:ordered) if(u==c){ found=true; break;} if(!found) ordered.push_back(c);} for(auto &k:ordered){ int cnt=0; for(auto &c:classes) if(c==k) cnt++; counts.push_back(cnt); keys.push_back(k);} std::vector<int> idx(ordered.size()); for(size_t i=0;i<idx.size();++i) idx[i]=i; std::stable_sort(idx.begin(), idx.end(), [&](int a,int b){ return counts[a] > counts[b]; }); std::vector<String> newOrdered; std::vector<int> newCounts; for(auto i:idx){ newOrdered.push_back(ordered[i]); newCounts.push_back(counts[i]); } ordered=newOrdered; counts=newCounts; keys=ordered; }
String makeDetectionPayload(const String &device, const std::vector<String> &ordered, const std::vector<int> &counts) { unsigned long ts = millis(); String p = String("{\"id\":\"") + device + String("\",\"classes\":["); for (size_t i = 0; i < ordered.size(); ++i) { p += "\""; p += ordered[i]; p += "\""; if (i + 1 < ordered.size()) p += ","; } p += "],\"counts\":{"; for (size_t i = 0; i < ordered.size(); ++i) { p += "\""; p += ordered[i]; p += "\":" + String(counts[i]); if (i + 1 < ordered.size()) p += ","; } p += "},\"ts\":" + String(ts); if (!ordered.empty()) p += ",\"topClass\":\"" + ordered[0] + "\""; p += "}"; return p; }
void publishDetectionPayload(const String &payload){ connectWiFi(); if(!client.connected()) connectMQTT(); if(!client.connected()){ Serial.println("MQTT not connected - skip detection publish"); return;} unsigned long now=millis(); if(payload==lastDetectionPayload && (now - lastDetectionPublishMs) < DETECTION_MIN_INTERVAL_MS) return; lastDetectionPublishMs = now; lastDetectionPayload = payload; String topic = detectionTopic(); client.beginMessage(topic.c_str(), payload.length(), true, 1); client.print(payload); client.endMessage(); Serial.println("Published detection (retained): " + topic + " -> " + payload); }

// --- setup / loop ---
void setup() {
  Serial.begin(115200);
  delay(100);

  // Water sensor ADC setup
  pinMode(WATER_SENSOR_PIN, INPUT); // ADC pins are inputs
  analogSetPinAttenuation(WATER_SENSOR_PIN, WATER_ADC_ATTEN);

  // Ultrasonic pins
  pinMode(ULTRASONIC_TRIG1, OUTPUT); pinMode(ULTRASONIC_ECHO1, INPUT);
  pinMode(ULTRASONIC_TRIG2, OUTPUT); pinMode(ULTRASONIC_ECHO2, INPUT);
  pinMode(ULTRASONIC_TRIG3, OUTPUT); pinMode(ULTRASONIC_ECHO3, INPUT);

  // Ensure trigs start LOW
  digitalWrite(ULTRASONIC_TRIG1, LOW);
  digitalWrite(ULTRASONIC_TRIG2, LOW);
  digitalWrite(ULTRASONIC_TRIG3, LOW);

  #ifdef SINGLE_PWM
    pinMode(PWM_PIN, OUTPUT);
    pinMode(DIR_PIN, OUTPUT);
  #else
    pinMode(RPWM, OUTPUT); pinMode(LPWM, OUTPUT);
  #endif
  pinMode(LEN, OUTPUT); pinMode(REN, OUTPUT);
  digitalWrite(LEN, HIGH); digitalWrite(REN, HIGH);

  // Buzzer pin init (added)
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW); // start off

  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);

  connectWiFi();
  net.setInsecure();
  connectMQTT();

  Serial.println("READY");
  Serial.println("Listening for detection lines on Serial (JSON or DETECTIONS:...)");
}

int collect_queue = 0;
int MAX_QUEUE = 8;
bool collecting = false;
bool detection_enabled = true;
unsigned long collect_sent_at = 0;
double last_local_collect_sent_at = 0.0;
const double LOCAL_ECHO_IGNORE_SECONDS = 1.0;
unsigned long last_log_time = 0;
String last_sent_summary = "";

void loop() {
  client.poll();
  unsigned long now = millis();

  // Serial input handling (from PC / detector)
  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if(line.length()==0) continue;
    Serial.println("RX: " + line);
    String up = line; up.toUpperCase(); up.trim();

    if(up == "COLLECT") {
      if(state != IDLE) {
        Serial.println("COLLECT ignored: motor busy");
      } else {
        Serial.println("COLLECTING (cmd received)");
        motorStart = now;
        motor_start_up();
        state = GOING_UP;
      }
      continue;
    } else if(up == "DONE") {
      Serial.println("done");
      motor_stop();
      state = IDLE;
      continue;
    }

    // --- PARSE DETECTIONS: queue them instead of immediate publish ---
    std::vector<String> classes = parseClasses(line);

    // If no classes were parsed, enqueue a "None" detection so we always publish something
    if (classes.empty()) {
      classes.push_back("None");
      Serial.println("No classes parsed from line -> forcing class \"None\"");
    }

    std::vector<String> ordered; 
    std::vector<int> counts; 
    std::vector<String> keys;
    buildOrderedCounts(classes, ordered, counts, keys);
    String payload = makeDetectionPayload(String(deviceId), ordered, counts);

    // Queue detection payload so it's published with sensor data block
    pendingDetectionPayload = payload;
    pendingDetection = true;
    Serial.println("Queued detection payload for next sensor publish: " + payload);
  }

  // Motor state machine (non-blocking)
  if (state == GOING_UP) {
    if (now - motorStart >= motorUP) {
      motor_stop();
      motorStart = now;
      state = WAIT_AFTER_UP;
      Serial.println("Reached top, idling 1s");
    }
  } else if (state == WAIT_AFTER_UP) {
    if (now - motorStart >= idleAfterUpMs) {
      motor_start_down();
      motorStart = now;
      state = GOING_DOWN;
      Serial.println("Start going down");
    }
  } else if (state == GOING_DOWN) {
    if (now - motorStart >= motorDOWN) {
      motor_stop();
      state = IDLE;
      Serial.println("done");
    }
  }

  // GPS read
  while (gpsSerial.available()) gps.encode(gpsSerial.read());

  // ---------- SENSOR & BUZZER: run when IDLE even if MQTT isn't connected ----------
  if (state == IDLE && (now - lastPubSensors >= SENSORS_INTERVAL)) {
    lastPubSensors = now;

    // --- Water sensor (ADC or digital) ---
    bool isADCpin = (WATER_SENSOR_PIN >= 34 && WATER_SENSOR_PIN <= 39);
    int analogVal = -1;
    bool rawHigh = false;
    if (isADCpin) {
      analogVal = analogRead(WATER_SENSOR_PIN);      // ~0..4095
      rawHigh = (analogVal >= WATER_THRESHOLD);
    } else {
      int dig = digitalRead(WATER_SENSOR_PIN);
      rawHigh = (dig == HIGH);
    }
    #if WATER_SENSOR_INVERTED
      bool rawState = rawHigh;
    #else
      bool rawState = !rawHigh;
    #endif

    // debounce
    if (rawState != floodStableState) {
      if (floodLastChangeMs == 0) floodLastChangeMs = now;
      else if (now - floodLastChangeMs >= FLOOD_DEBOUNCE_MS) {
        floodStableState = rawState;
        floodLastChangeMs = 0;
      }
    } else {
      floodLastChangeMs = 0;
    }
    bool flooded = floodStableState;

    // --- Ultrasonic read (robust, sequential to reduce crosstalk) ---
    // Tuning constants for ultrasonic robustness
    const float SENSOR_MIN_CM = 5.0f;            // increased blind zone to avoid rim/fast reflections
    const unsigned long INTER_SENSOR_DELAY_MS = 80; // pause between sensors to reduce crosstalk
    const int LOCAL_ATTEMPTS = 4;                // attempts per sensor for better averaging
    const float MAX_ACCEPTABLE_SPREAD = 20.0f;   // if sensors disagree more than this, bias safer value

    long d1 = readUltrasonicCM(ULTRASONIC_TRIG1, ULTRASONIC_ECHO1, LOCAL_ATTEMPTS);
    delay(INTER_SENSOR_DELAY_MS);
    long d2 = readUltrasonicCM(ULTRASONIC_TRIG2, ULTRASONIC_ECHO2, LOCAL_ATTEMPTS);
    delay(INTER_SENSOR_DELAY_MS);
    long d3 = readUltrasonicCM(ULTRASONIC_TRIG3, ULTRASONIC_ECHO3, LOCAL_ATTEMPTS);

    // Debug: print raw sensor values (helpful while tuning)
    Serial.print("RAW d1,d2,d3: ");
    Serial.print(d1); Serial.print(", ");
    Serial.print(d2); Serial.print(", ");
    Serial.println(d3);

    // Validate/filter readings (reject blind-zone and out-of-range)
    std::vector<float> valid;
    if (d1 > 0 && d1 >= SENSOR_MIN_CM && d1 <= ULTRASONIC_MOUNT_CM) valid.push_back((float)d1);
    if (d2 > 0 && d2 >= SENSOR_MIN_CM && d2 <= ULTRASONIC_MOUNT_CM) valid.push_back((float)d2);
    if (d3 > 0 && d3 >= SENSOR_MIN_CM && d3 <= ULTRASONIC_MOUNT_CM) valid.push_back((float)d3);

    // Choose aggregated distance robustly (median-like)
    float avgDist = -1.0f;
    if (!valid.empty()) {
      std::sort(valid.begin(), valid.end());
      if (valid.size() == 1) {
        avgDist = valid[0];
      } else if (valid.size() == 2) {
        avgDist = 0.5f * (valid[0] + valid[1]);
      } else {
        avgDist = valid[valid.size()/2]; // middle of 3
      }

      float spread = valid.back() - valid.front();
      if (spread > MAX_ACCEPTABLE_SPREAD) {
        // sensors disagree too much — bias toward the larger distance to avoid false "full"
        avgDist = valid.back();
      }
    } else {
      avgDist = -1.0f; // no valid reading
    }

    // use avgDist for binFull decision as well
    bool binFull = (avgDist > 0.0f && avgDist < (float)BIN_FULL_CM);

    // --- Compute fill percentage (whole number) based on mount height ---
    float fillPct = 0.0f;
    if (avgDist > 0.0f) {
      float effectiveHeight = ULTRASONIC_MOUNT_CM - SENSOR_MIN_CM;
      float adjustedDist = avgDist - SENSOR_MIN_CM;
      if (adjustedDist < 0) adjustedDist = 0;
      fillPct = 100.0f * (effectiveHeight - adjustedDist) / effectiveHeight;
      // Clamp between 0 and 100
      if (fillPct < 0) fillPct = 0;
      if (fillPct > 100) fillPct = 100;
    } else {
      // no valid reading -> treat as 0% (or set to -1 to indicate N/A)
      fillPct = 0.0f;
    }

    // --- Buzzer debounce logic: require consecutive confirmations ---
    static int consecutiveFulls = 0;
    const int CONSECUTIVE_FULLS_REQUIRED = 1; // set to 1 for testing; raise to 3 in production
    const float BUZZER_THRESHOLD = 90.0f;

    if (fillPct >= BUZZER_THRESHOLD) {
      consecutiveFulls++;
      Serial.print("consecutiveFulls="); Serial.print(consecutiveFulls);
      Serial.print(" fillPct="); Serial.println(fillPct);
      if (consecutiveFulls >= CONSECUTIVE_FULLS_REQUIRED) {
        // If your buzzer requires PWM, use tone(BUZZER_PIN, frequency) instead
        digitalWrite(BUZZER_PIN, HIGH);
      }
    } else {
      if (consecutiveFulls > 0) {
        Serial.print("consecutiveFulls reset (fillPct="); Serial.print(fillPct); Serial.println(")");
      }
      consecutiveFulls = 0;
      digitalWrite(BUZZER_PIN, LOW);
    }

    // debug prints
    Serial.print("WATER_RAW=");
    if (isADCpin) { Serial.print(analogVal); Serial.print(" (adc)"); }
    else Serial.print(rawHigh ? "HIGH" : "LOW");
    Serial.print(" rawState="); Serial.print(rawState ? "1" : "0");
    Serial.print(" stable="); Serial.println(flooded ? "true" : "false");

    Serial.print("avgDist=");
    if (avgDist > 0.0f) Serial.print(avgDist); else Serial.print("N/A");
    Serial.print("  Fill%=");
    Serial.println(fillPct);

    // publish flood/bin messages only if MQTT connected — but readings + buzzer happen regardless
    if (WiFi.status() == WL_CONNECTED && client.connected()) {
      // publish flood
      String floodPayload = String("{\"id\":\"") + deviceId + String("\",\"flooded\":") + (flooded ? "true" : "false") + "}";
      client.beginMessage(floodTopic);
      client.print(floodPayload);
      client.endMessage();
      Serial.println("Published Flood: " + floodPayload);

      // publish bin full (include fillPct)
      String binPayload = String("{\"id\":\"") + deviceId + String("\",\"binFull\":") + (binFull ? "true" : "false") + ",\"fillPct\":" + String(fillPct) + "}";
      client.beginMessage(binFullTopic);
      client.print(binPayload);
      client.endMessage();
      Serial.println("Published Bin Full: " + binPayload);

      // --- NEW: publish pending detection payload (if any) at the same time ---
      if (pendingDetection && pendingDetectionPayload.length() > 0) {
        unsigned long nowDet = millis();
        // respect detection debounce as before
        if (pendingDetectionPayload != lastDetectionPayload || (nowDet - lastDetectionPublishMs) >= DETECTION_MIN_INTERVAL_MS) {
          // publish via helper (handles connect + dedupe + retained)
          publishDetectionPayload(pendingDetectionPayload);
        } else {
          Serial.println("Pending detection suppressed due to debounce/duplicate.");
        }
        // clear queue no matter whether published or suppressed - it will be re-queued by the Pi later if needed
        pendingDetection = false;
        pendingDetectionPayload = "";
      }
    } else {
      Serial.println("MQTT not connected — readings taken locally, not published.");
    }

  } // end sensor block

  delay(10);
}
