
#include <Arduino.h>
#include <Preferences.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <ArduinoMqttClient.h>
#include <TinyGPSPlus.h>
#include <vector>
#include <WiFiManager.h> 
#include <HX711_ADC.h> 

#define RPWM 22
#define LPWM 21
#define REN  19
#define LEN  18
#define GPS_RX            16
#define GPS_TX            17
#define WATER_SENSOR_PIN  34
#define ULTRASONIC_TRIG1  33
#define ULTRASONIC_ECHO1  32
#define ULTRASONIC_TRIG2  27
#define ULTRASONIC_ECHO2  26
#define BUZZER_PIN        14
#define HX711_dout = 4;

#define HX711_sck  = 23;

Preferences prefs;

enum CalibTarget { CALIB_NONE = 0, CALIB_DRY = 1, CALIB_WET = 2 };
void startCalibration(CalibTarget target);
void calibrate(); 
void changeSavedCalFactor();
void interactiveLoadCellCalibration(); 
void runLoadCellCalibration(float known_mass_g);

// motor timing & speed 
unsigned long motorUP_ms   = 5000;
unsigned long motorDOWN_ms = 3000;
const unsigned long MAX_SAFE_MS = 120000UL;
const unsigned long IDLE_AFTER_UP_MS = 1000UL;

bool runningUp = false;
bool runningDown = false;
unsigned long runStartMs = 0;

const float BIN_FULL_KG = 8.0;      // bin considered full when measured weight >= this (kg)
const int   BIN_FULL_CM = 10;        // bin considered full when ultrasonic distance < this (cm)
const int   BIN_FULL_CONSECUTIVE = 3; // number of consecutive "full" samples to declare stable full
int motorSpeed = 255; 

const char* deviceId = "DVC006";
const char* ssid = "Test";
const char* wifi_pass = "1234567890";
const char* mqttBroker = "a62b022814fc473682be5d58d05e5f97.s1.eu.hivemq.cloud";
const int mqttPort = 8883;
const char* mqttUser = "prototype";
const char* mqttPassword = "Prototype1";
const char* floodTopic = "esp32/sensor/flood";
const char* binFullTopic = "esp32/sensor/bin_full";

String ultrasonic12Topic() { return String("esp32/") + deviceId + "/sensor/ultrasonic12"; }
String detectionTopic()   { return String("esp32/") + deviceId + "/sensor/detections"; }
String weightTopic()      { return String("esp32/") + deviceId + "/sensor/weight"; }

WiFiClientSecure net;
MqttClient client(net);
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);

// timing
unsigned long lastPubGPS = 0;
unsigned long lastPubSensors = 0;
const unsigned long GPS_INTERVAL = 5000;
const unsigned long SENSORS_INTERVAL = 2000;
const unsigned long DETECTION_MIN_INTERVAL_MS = 800;

const float SENSOR_MIN_CM = 5.0;    
const unsigned long PRESENCE_DEBOUNCE_MS = 1500;

// water sensor basic config
#define WATER_SENSOR_INVERTED 1
const adc_attenuation_t WATER_ADC_ATTEN = ADC_11db;

// --- Water smoothing & calibration globals ---
const int WATER_AVG_WINDOW = 8;       
int waterSamples[WATER_AVG_WINDOW];
int waterSampleIdx = 0;
bool waterSamplesInit = false;

bool lastFloodedState = false;

int WATER_THRESHOLD_WET = 550;  
int WATER_THRESHOLD_DRY = 500;  

bool USE_EMA = false;
float waterEMA = -1.0;
const float WATER_EMA_ALPHA = 0.25; // 0..1

CalibTarget calibTarget = CALIB_NONE;
bool calibrating = false;
unsigned long calibStartMs = 0;
const unsigned long calibDurationMs = 4000UL; // collect for 4s
long calibSum = 0;
int calibCount = 0;
int lastDryAvg = -1;
int lastWetAvg = -1;
// ------------------------------------------------------------------------------

const unsigned long BUZZER_DURATION_MS = 3000;
unsigned long buzzerOnUntilMs = 0;

bool pendingDetection = false;
String pendingDetectionPayload = "";
String lastDetectionPayload = "";
unsigned long lastDetectionPublishMs = 0;

bool lastDetectionPublishSuccess = false;
unsigned long lastDetectionSuccessMs = 0;

volatile bool collectRequested = false;

HX711_ADC LoadCell(HX711_dout, HX711_sck);
float load_cal_factor = 20.38; 
bool loadcell_ready = false;

bool suspendSensorOutput = false; 

bool tareInProgress = false;

// helpers
long readUltrasonicCM(uint8_t trigPin, uint8_t echoPin, int attempts = 3) {
  long sum = 0;
  int valid = 0;
  for (int i = 0; i < attempts; ++i) {
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    unsigned long duration = pulseIn(echoPin, HIGH, 30000UL); // 30ms timeout
    if (duration > 0) {
      long cm = (long)(duration / 29UL / 2UL);
      sum += cm;
      valid++;
    }
    delay(10);
  }
  if (valid == 0) return -1;
  return sum / valid;
}

bool publishDetectionPayload(const String &payload, bool force = false) {
  unsigned long now = millis();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("publishDetectionPayload: WiFi not connected");
    lastDetectionPublishSuccess = false;
    return false;
  }
  if (!client.connected()) {
    Serial.println("publishDetectionPayload: MQTT not connected");
    lastDetectionPublishSuccess = false;
    return false;
  }
  if (!force && payload == lastDetectionPayload && (now - lastDetectionPublishMs) < DETECTION_MIN_INTERVAL_MS) {
    Serial.println("publishDetectionPayload: suppressed duplicate (treated as already published)");
    lastDetectionPublishSuccess = true;
    lastDetectionSuccessMs = now;
    return true;
  }
  lastDetectionPayload = payload;
  lastDetectionPublishMs = now;
  String topic = detectionTopic();
  client.beginMessage(topic.c_str(), payload.length(), true, 1);
  client.print(payload);
  bool ok = client.endMessage();
  if (ok) {
    lastDetectionPublishSuccess = true;
    lastDetectionSuccessMs = now;
    Serial.println("Published detection: " + payload);
  } else {
    lastDetectionPublishSuccess = false;
    Serial.println("Failed to publish detection (endMessage returned false)");
  }
  return ok;
}

// --- CHANGED: publish weight in kg (payload uses weight_kg) ---
void publishWeightToMQTT(float weight_kg) {
  if (WiFi.status() != WL_CONNECTED || !client.connected()) return;
  String topic = weightTopic();
  String payload = String("{\"id\":\"") + deviceId + String("\",")
    + "\"weight_kg\":" + String(weight_kg, 3) + ","
    + "\"ts\":" + String(millis()) + "}";
  client.beginMessage(topic.c_str(), payload.length(), false, 1);
  client.print(payload);
  client.endMessage();
  if (!suspendSensorOutput) Serial.println("Published weight: " + payload);
}

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.begin(ssid, wifi_pass);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 10000) {
    delay(300);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWi-Fi connected: " + WiFi.localIP().toString());
  } else {
    Serial.println("\nWi-Fi failed");
  }
}

void connectMQTT() {
  if (client.connected()) return;
  if (WiFi.status() != WL_CONNECTED) return;
  client.setUsernamePassword(mqttUser, mqttPassword);
  String clientId = String("esp32-") + deviceId + "-" + String(random(0xffff), HEX);
  client.setId(clientId.c_str());
  Serial.print("Connecting MQTT...");
  if (client.connect(mqttBroker, mqttPort)) Serial.println("MQTT OK");
  else Serial.println("MQTT fail");
}

void publishUltrasonicBlock(float d1f, float d2f, bool stableBinFull) {
  if (WiFi.status() != WL_CONNECTED || !client.connected()) return;
  String topic = ultrasonic12Topic();
  String payload = String("{\"id\":\"") + deviceId + String("\",")
    + "\"d1\":" + String((long)(d1f < 0 ? -1 : (long)d1f)) + ","
    + "\"d2\":" + String((long)(d2f < 0 ? -1 : (long)d2f)) + ","
    + "\"ts\":" + String(millis()) + "}";
  client.beginMessage(topic.c_str(), payload.length(), false, 1);
  client.print(payload);
  client.endMessage();
  if (!suspendSensorOutput) Serial.println("Published ultrasonic: " + payload);

  String floodPayload = String("{\"id\":\"") + deviceId + String("\",\"flooded\":false}");
  client.beginMessage(floodTopic);
  client.print(floodPayload);
  client.endMessage();
  String binPayload = String("{\"id\":\"") + deviceId + String("\",\"binFull\":") + (stableBinFull ? "true" : "false") + "}";
  client.beginMessage(binFullTopic);
  client.print(binPayload);
  client.endMessage();
  if (!suspendSensorOutput) Serial.println("Published bin: " + binPayload);

  if (pendingDetection && pendingDetectionPayload.length() > 0) {
    bool ok = publishDetectionPayload(pendingDetectionPayload, false);
    if (ok) {
      pendingDetection = false;
      pendingDetectionPayload = "";
      if (!suspendSensorOutput) Serial.println("Pending detection published from publishUltrasonicBlock");
    } else {
      if (!suspendSensorOutput) Serial.println("Pending detection publish failed inside publishUltrasonicBlock (will retry later or on COLLECT)");
    }
  }
}

// --- Motor control helpers (merged) ---
void motor_start_up() {
  analogWrite(RPWM, motorSpeed);
  analogWrite(LPWM, 0);
  digitalWrite(LEN, HIGH);
  digitalWrite(REN, HIGH);
  if (!suspendSensorOutput) Serial.println("Motor START UP");
  runningUp = true;
  runningDown = false;
  runStartMs = millis();
}

void motor_start_down() {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, motorSpeed);
  digitalWrite(LEN, HIGH);
  digitalWrite(REN, HIGH);
  if (!suspendSensorOutput) Serial.println("Motor START DOWN");
  runningDown = true;
  runningUp = false;
  runStartMs = millis();
}

void motor_stop() {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);
  digitalWrite(LEN, HIGH);
  digitalWrite(REN, HIGH);
  if (!suspendSensorOutput) Serial.println("Motor STOP");
  runningUp = runningDown = false;
}

void updateMotorSpeed() {
  if (runningUp) {
    analogWrite(RPWM, motorSpeed);
    analogWrite(LPWM, 0);
  } else if (runningDown) {
    analogWrite(RPWM, 0);
    analogWrite(LPWM, motorSpeed);
  }
}

void motorStopAll() {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);
  digitalWrite(REN, HIGH);
  digitalWrite(LEN, HIGH);
  if (!suspendSensorOutput) Serial.println("Motor stopped (all)");
}

void motorUp(int speed) {
  digitalWrite(REN, HIGH);
  digitalWrite(LEN, HIGH);
  analogWrite(RPWM, speed);
  analogWrite(LPWM, 0);
  if (!suspendSensorOutput) Serial.println(String("Motor UP at speed ") + String(speed));
}

void motorDown(int speed) {
  digitalWrite(REN, HIGH);
  digitalWrite(LEN, HIGH);
  analogWrite(LPWM, speed);
  analogWrite(RPWM, 0);
  if (!suspendSensorOutput) Serial.println(String("Motor DOWN at speed ") + String(speed));
}

// Run collection sequence: up -> stop -> down while servicing MQTT & GPS
void runCollectionSequence(unsigned long upMs, unsigned long downMs, int speed) {
  // Start UP
  motorUp(speed);
  unsigned long endUp = millis() + upMs;
  while (millis() < endUp) {
    client.poll();
    while (gpsSerial.available()) gps.encode(gpsSerial.read());
    delay(10);
  }
  motorStopAll();

  // Pause briefly
  unsigned long endPause = millis() + IDLE_AFTER_UP_MS;
  while (millis() < endPause) {
    client.poll();
    while (gpsSerial.available()) gps.encode(gpsSerial.read());
    delay(10);
  }

  // Start DOWN
  motorDown(speed);
  unsigned long endDown = millis() + downMs;
  while (millis() < endDown) {
    client.poll();
    while (gpsSerial.available()) gps.encode(gpsSerial.read());
    delay(10);
  }
  motorStopAll();
}

// Perform collection actions and ensure collector gets pending detection, then send DONE over serial
void performCollectionActions() {
  if (!suspendSensorOutput) Serial.println("Performing collection actions (motor sequence)...");
  runCollectionSequence(motorUP_ms, motorDOWN_ms, motorSpeed);

  if (pendingDetection && pendingDetectionPayload.length() > 0) {
    bool ok = publishDetectionPayload(pendingDetectionPayload, true);
    if (ok) {
      pendingDetection = false;
      pendingDetectionPayload = "";
      if (!suspendSensorOutput) Serial.println("Pending detection published successfully during COLLECT");
    } else {
      if (!suspendSensorOutput) {
        Serial.println("Pending detection publish failed during COLLECT - printing payload to Serial for collector:");
        Serial.println(pendingDetectionPayload);
      }
    }
  } else {
    if (!suspendSensorOutput) Serial.println("No pending detection to publish during COLLECT");
  }

  // ALWAYS send DONE so external collector sees it even if sensor output suppressed
  Serial.println("DONE"); // important for external collector
  if (!suspendSensorOutput) Serial.println("Collection completed - DONE sent");
  collectRequested = false;
}

// --- Preferences & serial helpers from first sketch ---
void savePrefs() {
  prefs.begin("motorcal", false);
  prefs.putULong("up", motorUP_ms);
  prefs.putULong("down", motorDOWN_ms);
  prefs.putInt("speed", motorSpeed);
  // store water thresholds optionally
  prefs.putInt("w_wet", WATER_THRESHOLD_WET);
  prefs.putInt("w_dry", WATER_THRESHOLD_DRY);
  prefs.putBool("w_ema", USE_EMA);
  // store load cell calibration factor
  prefs.putFloat("load_cal", load_cal_factor);
  prefs.end();
  Serial.println(F("Saved calibration to flash."));
}

void loadPrefs() {
  prefs.begin("motorcal", true);
  motorUP_ms = prefs.getULong("up", motorUP_ms);
  motorDOWN_ms = prefs.getULong("down", motorDOWN_ms);
  motorSpeed = prefs.getInt("speed", motorSpeed);
  WATER_THRESHOLD_WET = prefs.getInt("w_wet", WATER_THRESHOLD_WET);
  WATER_THRESHOLD_DRY = prefs.getInt("w_dry", WATER_THRESHOLD_DRY);
  USE_EMA = prefs.getBool("w_ema", USE_EMA);
  load_cal_factor = prefs.getFloat("load_cal", load_cal_factor);
  prefs.end();
  Serial.println(F("Loaded calibration from flash."));
}

void printHelp() {
  Serial.println(F("--- Motor Calibrator Help ---"));
  Serial.println(F("Commands:"));
  Serial.println(F("  help             - show this help"));
  Serial.println(F("  u                - START motor UP (press U to stop)"));
  Serial.println(F("  U                - STOP motor UP"));
  Serial.println(F("  d                - START motor DOWN (press D to stop)"));
  Serial.println(F("  D                - STOP motor DOWN"));
  Serial.println(F("  set speed <val>  - change speed (0–255) in real time"));
  Serial.println(F("  set up <ms>      - set motorUP time"));
  Serial.println(F("  set down <ms>    - set motorDOWN time"));
  Serial.println(F("  save             - save calibration (including water thresholds & load cal)"));
  Serial.println(F("  load             - load calibration"));
  Serial.println(F("  status           - show settings"));
  Serial.println(F("  run              - run full cycle"));
  Serial.println(F("  test up <ms>     - test UP for given ms"));
  Serial.println(F("  test down <ms>   - test DOWN for given ms"));
  Serial.println(F("  COLLECT          - run collection sequence (used by external collector)"));
  Serial.println(F(""));
  Serial.println(F("Water calibration commands:"));
  Serial.println(F("  calibdry         - collect samples for dry state (4s)"));
  Serial.println(F("  calibwet         - collect samples for wet state (4s)"));
  Serial.println(F("  useema on|off    - enable/disable exponential moving average"));
  Serial.println(F(""));
  Serial.println(F("Load cell commands:"));
  Serial.println(F("  lt               - tare load cell (no delay)"));
  Serial.println(F("  lcal <mass>      - calibrate using known mass (mass in grams) and SAVE"));
  Serial.println(F("  lcal             - interactive calibrate (prompts for tare and known mass)"));
  Serial.println(F("  lset <cal>       - set cal factor directly and SAVE"));
  Serial.println(F("  lload            - reload cal factor from prefs"));
  Serial.println(F("  lshow            - show current cal factor and last reading"));
  Serial.println(F("------------------------------"));
}

void printStatus() {
  Serial.print(F("motorUP_ms = ")); Serial.println(motorUP_ms);
  Serial.print(F("motorDOWN_ms = ")); Serial.println(motorDOWN_ms);
  Serial.print(F("motorSpeed = ")); Serial.println(motorSpeed);
  Serial.print(F("Water thresholds (wet/dry) = ")); Serial.print(WATER_THRESHOLD_WET); Serial.print("/"); Serial.println(WATER_THRESHOLD_DRY);
  Serial.print(F("USE_EMA = ")); Serial.println(USE_EMA ? "YES":"NO");
  Serial.print(F("Load cell cal factor = ")); Serial.println(load_cal_factor, 6);
}

void splitTokens(const String &line, std::vector<String> &tokens) {
  tokens.clear();
  int i = 0;
  while (i < (int)line.length()) {
    while (i < (int)line.length() && isspace(line[i])) i++;
    if (i >= (int)line.length()) break;
    int j = i;
    while (j < (int)line.length() && !isspace(line[j])) j++;
    tokens.push_back(line.substring(i, j));
    i = j;
  }
}

void printStatusToSerial() {
  Serial.println("=== STATUS ===");
  Serial.print("MQTT connected: "); Serial.println(client.connected() ? "YES" : "NO");
  Serial.print("Pending detection: "); Serial.println(pendingDetection ? "YES" : "NO");
  if (pendingDetection) {
    Serial.print("Pending payload: "); Serial.println(pendingDetectionPayload);
  }
  Serial.print("Last detection publish success: "); Serial.println(lastDetectionPublishSuccess ? "YES" : "NO");
  if (lastDetectionPublishSuccess) {
    Serial.print("Last success ms: "); Serial.println(lastDetectionSuccessMs);
  }
  Serial.print("Last detection payload (last attempted): "); Serial.println(lastDetectionPayload);
  Serial.print("Last detection publish attempt ms: "); Serial.println(lastDetectionPublishMs);
  Serial.print("Load cell cal factor: "); Serial.println(load_cal_factor, 6);
  Serial.println("=== END STATUS ===");
}

int binFullCounter = 0;

// ----------------- Water helper implementations -----------------
void waterInitBuffer() {
  for (int i = 0; i < WATER_AVG_WINDOW; ++i) waterSamples[i] = analogRead(WATER_SENSOR_PIN);
  waterSampleIdx = 0;
  waterSamplesInit = true;
}

int readWaterAvgOnce() {
  int v = analogRead(WATER_SENSOR_PIN);
  waterSamples[waterSampleIdx] = v;
  waterSampleIdx = (waterSampleIdx + 1) % WATER_AVG_WINDOW;

  long sum = 0;
  for (int i = 0; i < WATER_AVG_WINDOW; ++i) sum += waterSamples[i];
  return (int)(sum / WATER_AVG_WINDOW);
}

float updateWaterEMA(int raw) {
  if (waterEMA < 0.0) {
    waterEMA = (float)raw;
  } else {
    waterEMA = (1.0 - WATER_EMA_ALPHA) * waterEMA + WATER_EMA_ALPHA * (float)raw;
  }
  return waterEMA;
}

void startCalibration(CalibTarget target) {
  calibTarget = target;
  calibrating = true;
  calibStartMs = millis();
  calibSum = 0;
  calibCount = 0;
  suspendSensorOutput = true; // suppress noisy telemetry while we collect calibration samples
  Serial.println("\n===== CALIBRATION STARTED: " + String(target == CALIB_DRY ? "DRY" : "WET") + " =====");
  Serial.println("(sensor telemetry suppressed until calibration completes)");
}

void updateCalibration() {
  if (!calibrating || calibTarget == CALIB_NONE) return;
  int raw = analogRead(WATER_SENSOR_PIN);
  calibSum += raw;
  calibCount++;
  if (millis() - calibStartMs >= calibDurationMs) {
    float avg = (calibCount > 0) ? (float)calibSum / (float)calibCount : 0.0;
    Serial.print("Calibration sample average = ");
    Serial.println(avg, 1);
    if (calibTarget == CALIB_DRY) {
      lastDryAvg = (int)(avg + 0.5);
      Serial.print("Recorded dry average = "); Serial.println(lastDryAvg);
    } else if (calibTarget == CALIB_WET) {
      lastWetAvg = (int)(avg + 0.5);
      Serial.print("Recorded wet average = "); Serial.println(lastWetAvg);
    }
    calibrating = false;
    calibTarget = CALIB_NONE;

    // If both dry and wet are present, suggest thresholds and apply
    if (lastDryAvg >= 0 && lastWetAvg >= 0) {
      int midpoint = (lastDryAvg + lastWetAvg) / 2;
      int hysteresisGap = max(20, abs(lastWetAvg - lastDryAvg) / 6); // small fraction
      int suggestedWet = midpoint + hysteresisGap/2;
      int suggestedDry = midpoint - hysteresisGap/2;
      Serial.println("Suggested thresholds:");
      Serial.print("  WATER_THRESHOLD_WET = "); Serial.println(suggestedWet);
      Serial.print("  WATER_THRESHOLD_DRY = "); Serial.println(suggestedDry);
      // auto-apply suggested thresholds
      WATER_THRESHOLD_WET = suggestedWet;
      WATER_THRESHOLD_DRY = suggestedDry;
      Serial.println("Applied suggested thresholds.");
    } else {
      Serial.println("You must run both calibdry and calibwet to generate suggested thresholds.");
    }

    // signal calibration finished and resume regular telemetry
    suspendSensorOutput = false;
    Serial.println("===== CALIBRATION COMPLETE =====\n");
  }
}

bool computeFloodedWithHysteresis(int avgVal, float emaVal = -1.0) {
  int valToCheck = avgVal;
  if (USE_EMA && emaVal >= 0.0) {
    valToCheck = (int)(emaVal + 0.5);
  }
#if WATER_SENSOR_INVERTED
  if (valToCheck >= WATER_THRESHOLD_WET) return true;
  if (valToCheck <= WATER_THRESHOLD_DRY) return false;
  return lastFloodedState;
#else
  if (valToCheck <= WATER_THRESHOLD_WET) return true;
  if (valToCheck >= WATER_THRESHOLD_DRY) return false;
  return lastFloodedState;
#endif
}

// ------------ Load cell initialization & utility ---------------
void initLoadCell() {
  Serial.println("Initializing load cell (HX711)...");
  LoadCell.begin();

  unsigned long stabilizingtime = 2000; // ms
  boolean _tare = true; // auto-tare on start

  // Start the HX711 (this schedules stabilisation + optional tare)
  LoadCell.start(stabilizingtime, _tare);

  // Wait for the HX711 library to finish its first update cycle (non-blocking loop)
  unsigned long t0 = millis();
  const unsigned long STARTUP_TIMEOUT_MS = 4000UL;
  bool updateOk = false;
  while (millis() - t0 < STARTUP_TIMEOUT_MS) {
    // LoadCell.update() returns true when a new reading is available
    if (LoadCell.update()) {
      updateOk = true;
      break;
    }
    delay(5);
  }

  // Check library flags only after allowing update() to run
  if (!updateOk) {
    Serial.println("LoadCell: update() did not return within timeout. Check wiring/power.");
    loadcell_ready = false;
    return;
  }

  // Now check for timeout / signal errors
  if (LoadCell.getTareTimeoutFlag()) {
    Serial.println("LoadCell: Tare timeout flag set (tare failed).");
    loadcell_ready = false;
    return;
  }
  if (LoadCell.getSignalTimeoutFlag()) {
    Serial.println("LoadCell: Signal timeout flag set (no signal). Check wiring & power.");
    loadcell_ready = false;
    return;
  }

  // OK — read cal factor from prefs and apply
  prefs.begin("motorcal", true);
  load_cal_factor = prefs.getFloat("load_cal", load_cal_factor);
  prefs.end();
  LoadCell.setCalFactor(load_cal_factor);
  loadcell_ready = true;
  Serial.print("LoadCell initialized. Cal factor = ");
  Serial.println(load_cal_factor, 6);
}

// Run automatic calibration using known mass (in grams)
void runLoadCellCalibration(float known_mass_g) {
  if (!loadcell_ready) {
    Serial.println("Load cell not ready. Call init or check wiring.");
    return;
  }

  Serial.println("\n===== LOADCELL CALIBRATION STARTED (non-interactive) =====");
  suspendSensorOutput = true; // quiet telemetry while calibrating

  Serial.println("Refreshing dataset and computing new calibration...");
  LoadCell.refreshDataSet();
  float newCal = LoadCell.getNewCalibration(known_mass_g);
  if (newCal <= 0.0) {
    Serial.println("Calibration failed (newCal <= 0). Keep existing value.");
    suspendSensorOutput = false;
    Serial.println("===== LOADCELL CALIBRATION FAILED =====\n");
    return;
  }
  load_cal_factor = newCal;
  LoadCell.setCalFactor(load_cal_factor);
  // save automatically to prefs
  prefs.begin("motorcal", false);
  prefs.putFloat("load_cal", load_cal_factor);
  prefs.end();
  Serial.print("New calibration factor set and saved: ");
  Serial.println(load_cal_factor, 6);

  suspendSensorOutput = false;
  Serial.println("===== LOADCELL CALIBRATION COMPLETE =====\n");
}

// ---------- Interactive calibration adapted from user-provided example ----------
void calibrate() {
  if (!loadcell_ready) {
    Serial.println("Load cell not ready. Call init or check wiring.");
    return;
  }

  Serial.println("***");
  Serial.println("Start calibration:");
  Serial.println("Place the load cell on a level stable surface.");
  Serial.println("Remove any load applied to the load cell.");
  Serial.println("Send 'lt' from serial monitor to set the tare offset.");

  boolean _resume = false;
  while (_resume == false) {
    LoadCell.update();
    if (Serial.available() > 0) {
      if (Serial.available() > 0) {
        char inByte = Serial.read();
        if (inByte == 't') {
          LoadCell.tareNoDelay();
          Serial.println("Tare requested (no delay). Waiting for tare to complete...");
        }
      }
    }
    if (LoadCell.getTareStatus() == true) {
      Serial.println("Tare complete");
      _resume = true;
    }
    delay(20);
  }

  Serial.println("Now, place your known mass on the loadcell.");
  Serial.println("Then send the weight of this mass (i.e. 100.0) from serial monitor.");

  float known_mass = 0;
  _resume = false;
  while (_resume == false) {
    LoadCell.update();
    if (Serial.available() > 0) {
      known_mass = Serial.parseFloat();
      if (known_mass != 0) {
        Serial.print("Known mass is: ");
        Serial.println(known_mass);
        _resume = true;
      }
    }
    delay(20);
  }

  LoadCell.refreshDataSet(); //refresh the dataset to be sure that the known mass is measured correct
  float newCalibrationValue = LoadCell.getNewCalibration(known_mass); //get the new calibration value

  Serial.print("New calibration value has been set to: ");
  Serial.print(newCalibrationValue, 6);
  Serial.println(", use this as calibration value (calFactor) in your project sketch.");
  Serial.println("Save this value to preferences? y/n");

  _resume = false;
  while (_resume == false) {
    if (Serial.available() > 0) {
      char inByte = Serial.read();
      if (inByte == 'y') {
        prefs.begin("motorcal", false);
        prefs.putFloat("load_cal", newCalibrationValue);
        prefs.end();
        // apply to LoadCell
        load_cal_factor = newCalibrationValue;
        LoadCell.setCalFactor(load_cal_factor);
        Serial.print("Value ");
        Serial.print(newCalibrationValue, 6);
        Serial.println(" saved to preferences.");
        _resume = true;
      }
      else if (inByte == 'n') {
        Serial.println("Value not saved to preferences");
        _resume = true;
      }
    }
    delay(20);
  }

  Serial.println("End calibration");
  Serial.println("***");
  Serial.println("To re-calibrate, send 'r' from serial monitor.");
  Serial.println("For manual edit of the calibration value, send 'c' from serial monitor.");
  Serial.println("***");
}

// manual edit of calibration factor (adapted to Preferences)
void changeSavedCalFactor() {
  if (!loadcell_ready) {
    Serial.println("Load cell not ready. Call init or check wiring.");
    return;
  }
  float oldCalibrationValue = LoadCell.getCalFactor();
  boolean _resume = false;
  Serial.println("***");
  Serial.print("Current value is: ");
  Serial.println(oldCalibrationValue, 6);
  Serial.println("Now, send the new value from serial monitor, i.e. 696.0");
  float newCalibrationValue = 0.0;
  while (_resume == false) {
    if (Serial.available() > 0) {
      newCalibrationValue = Serial.parseFloat();
      if (newCalibrationValue != 0) {
        Serial.print("New calibration value is: ");
        Serial.println(newCalibrationValue, 6);
        LoadCell.setCalFactor(newCalibrationValue);
        load_cal_factor = newCalibrationValue;
        _resume = true;
      }
    }
    delay(20);
  }
  _resume = false;
  Serial.print("Save this value to preferences? y/n");
  while (_resume == false) {
    if (Serial.available() > 0) {
      char inByte = Serial.read();
      if (inByte == 'y') {
        prefs.begin("motorcal", false);
        prefs.putFloat("load_cal", newCalibrationValue);
        prefs.end();
        Serial.print("Value ");
        Serial.print(newCalibrationValue, 6);
        Serial.println(" saved to preferences.");
        _resume = true;
      }
      else if (inByte == 'n') {
        Serial.println("Value not saved to preferences");
        _resume = true;
      }
    }
    delay(20);
  }
  Serial.println("End change calibration value");
  Serial.println("***");
}

// -------------------------------------------------------------------


// ----------------- setup / loop -----------------
void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println(F("Combined ESP32 Motor Calibrator + Ultrasonic/MQTT (sensors 1 & 2 only, no fillPct) + HX711"));

  // pins
  pinMode(ULTRASONIC_TRIG1, OUTPUT); pinMode(ULTRASONIC_ECHO1, INPUT);
  pinMode(ULTRASONIC_TRIG2, OUTPUT); pinMode(ULTRASONIC_ECHO2, INPUT);
  digitalWrite(ULTRASONIC_TRIG1, LOW);
  digitalWrite(ULTRASONIC_TRIG2, LOW);

  pinMode(BUZZER_PIN, OUTPUT); digitalWrite(BUZZER_PIN, LOW);

  // motor pins setup
  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);
  pinMode(REN, OUTPUT);
  pinMode(LEN, OUTPUT);
  motorStopAll();

  // water ADC
  pinMode(WATER_SENSOR_PIN, INPUT);
  analogSetPinAttenuation(WATER_SENSOR_PIN, WATER_ADC_ATTEN);

  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);

  loadPrefs();
  waterInitBuffer(); // initialize water buffer using a few samples
  printHelp();
  printStatus();

  // init load cell
  initLoadCell();

  // ------------------ WiFi provisioning: WiFiManager (easiest) ------------------
  {
    WiFiManager wm;
    wm.setTimeout(180); // seconds for config portal (3 minutes)

    Serial.println("Starting WiFiManager autoConnect (AP if no saved credentials)...");
    if (!wm.autoConnect("ESP32-ConfigAP")) {
      Serial.println("WiFiManager config portal timeout or failed. Trying fallback credentials...");
      connectWiFi(); // fallback to hard-coded ssid/wifi_pass (Test / 1234567890)
    } else {
      Serial.println("WiFiManager connected to Wi-Fi.");
    }
  }
  // ------------------------------------------------------------------------------

  net.setInsecure();
  connectMQTT();

  Serial.println("Setup complete");
}

void loop() {
  client.poll();
  unsigned long now = millis();

  // --- New: monitor tareInProgress and resume telemetry when done ---
  if (tareInProgress) {
    // keep HX711 updating so tare can finish
    LoadCell.update();
    if (LoadCell.getTareStatus() == true) {
      tareInProgress = false;
      suspendSensorOutput = false;
      Serial.println("Tare complete. Resuming sensor telemetry.");
    }
  }

  // --- Serial handling (merged) ---
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      if (!suspendSensorOutput) Serial.println(String("RX serial: ") + line);
      // quick COLLECT / STATUS checks (case-insensitive)
      if (line.equalsIgnoreCase("COLLECT")) {
        performCollectionActions();
      } else if (line.equalsIgnoreCase("STATUS")) {
        printStatusToSerial();
      } else if (line.equalsIgnoreCase("calibdry")) {
        startCalibration(CALIB_DRY);
      } else if (line.equalsIgnoreCase("calibwet")) {
        startCalibration(CALIB_WET);
      } else if (line.equalsIgnoreCase("help")) {
        printHelp();
      } else if (line.equalsIgnoreCase("useema on")) {
        USE_EMA = true; if (!suspendSensorOutput) Serial.println("USE_EMA = ON");
      } else if (line.equalsIgnoreCase("useema off")) {
        USE_EMA = false; if (!suspendSensorOutput) Serial.println("USE_EMA = OFF");
      } else {
        // tokenized commands (motor calibrator commands + tests + loadcell)
        std::vector<String> tokens;
        splitTokens(line, tokens);
        if (tokens.size() == 0) { /* nothing */ }
        else {
          String cmd = tokens[0];
          cmd.toLowerCase();

          if (cmd == "u") {
            motor_start_up();
          } else if (tokens[0] == "U") {
            if (runningUp) {
              motor_stop();
              unsigned long dur = millis() - runStartMs;
              runningUp = false;
              if (!suspendSensorOutput) { Serial.print(F("STOP UP. Duration: ")); Serial.println(dur); }
            }
          } else if (cmd == "d") {
            motor_start_down();
          } else if (tokens[0] == "D") {
            if (runningDown) {
              motor_stop();
              unsigned long dur = millis() - runStartMs;
              runningDown = false;
              if (!suspendSensorOutput) { Serial.print(F("STOP DOWN. Duration: ")); Serial.println(dur); }
            }
          } else if (cmd == "stop") {
            motor_stop();
            runningUp = runningDown = false;
            if (!suspendSensorOutput) Serial.println(F("STOP executed."));
          } else if (cmd == "set") {
            if (tokens.size() >= 3) {
              String which = tokens[1]; which.toLowerCase();
              if (which == "up") {
                motorUP_ms = tokens[2].toInt();
                if (!suspendSensorOutput) { Serial.print(F("motorUP_ms = ")); Serial.println(motorUP_ms); }
              } else if (which == "down") {
                motorDOWN_ms = tokens[2].toInt();
                if (!suspendSensorOutput) { Serial.print(F("motorDOWN_ms = ")); Serial.println(motorDOWN_ms); }
              } else if (which == "speed") {
                motorSpeed = tokens[2].toInt();
                if (motorSpeed < 0) motorSpeed = 0;
                if (motorSpeed > 255) motorSpeed = 255;
                if (!suspendSensorOutput) { Serial.print(F("motorSpeed updated = ")); Serial.println(motorSpeed); }
                updateMotorSpeed(); // apply new speed immediately
              }
            }
          } else if (cmd == "save") {
            savePrefs();
          } else if (cmd == "load") {
            loadPrefs();
            // re-apply load cal to loadcell if present
            if (loadcell_ready) {
              LoadCell.setCalFactor(load_cal_factor);
            }
            printStatus();
          } else if (cmd == "status") {
            printStatus();
          } else if (cmd == "run") {
            if (!suspendSensorOutput) Serial.println(F("Running full cycle"));
            motor_start_up();
            unsigned long start = millis();
            while (millis() - start < motorUP_ms) {
              updateMotorSpeed(); // allow real-time speed changes
              delay(10);
            }
            motor_stop();
            delay(IDLE_AFTER_UP_MS);
            motor_start_down();
            start = millis();
            while (millis() - start < motorDOWN_ms) {
              updateMotorSpeed();
              delay(10);
            }
            motor_stop();
          } else if (cmd == "test") {
            if (tokens.size() >= 3) {
              String which = tokens[1]; which.toLowerCase();
              unsigned long val = tokens[2].toInt();
              if (which == "up") {
                if (!suspendSensorOutput) { Serial.print(F("Test UP for ")); Serial.println(val); }
                motor_start_up();
                unsigned long s = millis();
                while (millis() - s < val) {
                  updateMotorSpeed();
                  delay(10);
                }
                motor_stop();
              } else if (which == "down") {
                if (!suspendSensorOutput) { Serial.print(F("Test DOWN for ")); Serial.println(val); }
                motor_start_down();
                unsigned long s = millis();
                while (millis() - s < val) {
                  updateMotorSpeed();
                  delay(10);
                }
                motor_stop();
              }
            }
          } else if (cmd == "lt") {
            // tare load cell
            if (loadcell_ready) {
              Serial.println("Load cell tare started (no delay). Suppressing sensor telemetry until tare completes.");
              LoadCell.tareNoDelay();
              tareInProgress = true;
              suspendSensorOutput = true; // stop noisy telemetry until tare finishes
            } else {
              Serial.println("Load cell not ready.");
            }
          } else if (cmd == "lcal") {
            // two behaviors:
            // - lcal <mass> -> non-interactive (existing)
            // - lcal        -> interactive (adapted)
            if (tokens.size() >= 2) {
              // calibrate using known mass (grams)
              if (loadcell_ready) {
                float known = tokens[1].toFloat();
                if (known <= 0.0) {
                  Serial.println("Invalid known mass for lcal. Use grams > 0.");
                } else {
                  runLoadCellCalibration(known);
                }
              } else {
                Serial.println("Load cell not ready.");
              }
            } else {
              // start interactive procedure (adapted example)
              calibrate();
            }
          } else if (cmd == "lset" && tokens.size() >= 2) {
            // set calibration factor directly and save
            float c = tokens[1].toFloat();
            if (c > 0.0) {
              load_cal_factor = c;
              if (loadcell_ready) LoadCell.setCalFactor(load_cal_factor);
              prefs.begin("motorcal", false);
              prefs.putFloat("load_cal", load_cal_factor);
              prefs.end();
              if (!suspendSensorOutput) {
                Serial.print("Calibration factor set and saved: ");
                Serial.println(load_cal_factor, 6);
              }
            } else {
              Serial.println("Invalid calibration factor.");
            }
          } else if (cmd == "lload") {
            prefs.begin("motorcal", true);
            load_cal_factor = prefs.getFloat("load_cal", load_cal_factor);
            prefs.end();
            if (loadcell_ready) LoadCell.setCalFactor(load_cal_factor);
            Serial.print("Loaded calibration factor: ");
            Serial.println(load_cal_factor, 6);
          } else if (cmd == "lshow") {
            float last_g = 0.0;
            if (loadcell_ready) {
              LoadCell.update();
              last_g = LoadCell.getData();
            }
            float last_kg = last_g / 1000.0f;
            Serial.print("Cal factor: "); Serial.println(load_cal_factor, 6);
            Serial.print("Last reading (kg): "); Serial.println(last_kg, 3);
          } else {
            // treat as possible JSON detection or unrecognized
            const char firstChar = line.charAt(0);
            if (firstChar == '{' || firstChar == '[') {
              pendingDetectionPayload = line;
              pendingDetection = true;
              if (!suspendSensorOutput) Serial.println("Received detection JSON over serial; stored pendingDetection");
              if (WiFi.status() == WL_CONNECTED && client.connected()) {
                bool ok = publishDetectionPayload(pendingDetectionPayload, false);
                if (ok) {
                  pendingDetection = false;
                  pendingDetectionPayload = "";
                  if (!suspendSensorOutput) Serial.println("Immediate publish of serial-provided detection succeeded");
                } else {
                  if (!suspendSensorOutput) Serial.println("Immediate publish of serial-provided detection failed (will retry later)");
                }
              } else {
                if (!suspendSensorOutput) Serial.println("WiFi/MQTT not connected - will publish pending detection later");
              }
            } else {
              if (!suspendSensorOutput) Serial.println(String("RX (unrecognized): ") + line);
            }
          }
        }
      }
    }
  }

  // Safety watchdog for manual runs
  if (runningUp || runningDown) {
    if (millis() - runStartMs > MAX_SAFE_MS) {
      motor_stop();
      runningUp = runningDown = false;
      if (!suspendSensorOutput) Serial.println(F("SAFETY STOP"));
    }
  }

  // Update motor PWM if running
  updateMotorSpeed();

  // GPS parsing & minimal publish
  while (gpsSerial.available()) gps.encode(gpsSerial.read());
  if (gps.location.isValid() && (gps.location.isUpdated() || now - lastPubGPS >= GPS_INTERVAL)) {
    lastPubGPS = now;
    if (WiFi.status() == WL_CONNECTED && client.connected()) {
      String t = String("esp32/") + deviceId + "/gps";
      String p = String("{\"id\":\"") + deviceId + String("\",\"lat\":") + String(gps.location.lat(),6)
                 + String(",\"lon\":") + String(gps.location.lng(),6) + String("}");
      client.beginMessage(t.c_str(), p.length(), false, 1);
      client.print(p);
      client.endMessage();
      if (!suspendSensorOutput) Serial.println("Published GPS: " + p);
    }
  }

  // sensor publishing block (every SENSORS_INTERVAL)
  if (now - lastPubSensors >= SENSORS_INTERVAL) {
    lastPubSensors = now;

    // ---------- Water read with smoothing + hysteresis ----------
    if (!waterSamplesInit) waterInitBuffer();

    int rawVal = analogRead(WATER_SENSOR_PIN);        // instantaneous raw ADC
    int avgVal = readWaterAvgOnce();                  // moving average
    float emaVal = updateWaterEMA(rawVal);           // EMA (if enabled)
    updateCalibration();                              // if calibrating, collect samples

    bool floodedNow = computeFloodedWithHysteresis(avgVal, (USE_EMA ? emaVal : -1.0));

    // print only on state change (or once per sample for tuning)
    if (!suspendSensorOutput) {
      if (floodedNow != lastFloodedState) {
        Serial.print("WATER state change: raw=");
        Serial.print(rawVal);
        Serial.print(" avg=");
        Serial.print(avgVal);
        if (USE_EMA) { Serial.print(" ema="); Serial.print(emaVal,1); }
        Serial.print(" flooded="); Serial.println(floodedNow ? "1":"0");
        lastFloodedState = floodedNow;
      } else {
        // occasional print for tuning — comment out to reduce spam
        Serial.print("WATER adc=");
        Serial.print(rawVal);
        Serial.print(" avg=");
        Serial.print(avgVal);
        if (USE_EMA) { Serial.print(" ema="); Serial.print(emaVal,1); }
        Serial.print(" flooded="); Serial.println(floodedNow ? "1":"0");
      }
    }

    bool flooded = floodedNow;

    // ---------- READ LOAD CELL FIRST (so binFull can be based on weight) ----------
    float weight_kg = 0.0f;
    if (loadcell_ready) {
      unsigned long t0 = millis();
      const unsigned long HX711_READ_TIMEOUT = 50UL; // ms
      bool got = false;

      // Try briefly to get an update, but don't block the loop
      while (millis() - t0 < HX711_READ_TIMEOUT) {
        if (LoadCell.update()) { got = true; break; }
        // small yield so other interrupts/tasks run
        delay(1);
      }

      // Get the (calibrated) data in grams (library expects calibration in g)
      float weight_g = LoadCell.getData();

      if (!got) {
        // still read last known value or 0 — avoid blocking
        if (!suspendSensorOutput) Serial.println("Warning: LoadCell.update() timeout (no new sample)");
      }

      // convert to kilograms for display/publish
      weight_kg = weight_g / 1000.0f;

      // *** CLAMP NEGATIVE OR INVALID WEIGHT TO ZERO (per your request) ***
      // if weight_kg is NaN or negative, replace with 0.0
      if (!(weight_kg >= 0.0f)) {
        weight_kg = 0.0f;
      }

      if (!suspendSensorOutput) {
        Serial.print("Load cell reading (kg): ");
        Serial.println(weight_kg, 3); // 3 decimal places
      }

      // publish to MQTT (now in kg)
      if (WiFi.status() == WL_CONNECTED && client.connected()) {
        publishWeightToMQTT(weight_kg);
      }
    }

    // ---------- Ultrasonic reads (for publishing and fallback) ----------
    long d1 = readUltrasonicCM(ULTRASONIC_TRIG1, ULTRASONIC_ECHO1);
    delay(60);
    long d2 = readUltrasonicCM(ULTRASONIC_TRIG2, ULTRASONIC_ECHO2);

    if (!suspendSensorOutput) {
      Serial.print("RAW d1,d2: ");
      Serial.print(d1); Serial.print(", "); Serial.println(d2);
    }

    bool rawBinFull = false;
    if (loadcell_ready) {
      rawBinFull = (weight_kg >= BIN_FULL_KG);
      if (!suspendSensorOutput) {
        Serial.print("Bin fullness based on weight (kg): "); Serial.println(weight_kg, 3);
        Serial.print("rawBinFull (weight >= "); Serial.print(BIN_FULL_KG); Serial.print(") = "); Serial.println(rawBinFull ? "1" : "0");
      }
    } else{
      // fallback to ultrasonic if no load cell
      float min12 = -1.0;
      if (d1 > 0 && d1 >= SENSOR_MIN_CM) min12 = (min12 < 0 ? d1 : min(min12, (float)d1));
      if (d2 > 0 && d2 >= SENSOR_MIN_CM) min12 = (min12 < 0 ? d2 : min(min12, (float)d2));
      if (min12 >= 0) rawBinFull = (min12 < (float)BIN_FULL_CM);
      if (!suspendSensorOutput) {
        Serial.print("Bin fullness based on ultrasonic min12: "); Serial.println(min12);
        Serial.print("rawBinFull (ultrasonic < "); Serial.print(BIN_FULL_CM); Serial.print(") = "); Serial.println(rawBinFull ? "1" : "0");
      }
    }

    if (rawBinFull) {
      if (binFullCounter < BIN_FULL_CONSECUTIVE) ++binFullCounter;
    } else {
      if (binFullCounter > 0) --binFullCounter;
    }
    bool stableBinFull = (binFullCounter >= BIN_FULL_CONSECUTIVE);

    if (stableBinFull && buzzerOnUntilMs == 0) {
      digitalWrite(BUZZER_PIN, HIGH);
      buzzerOnUntilMs = now + BUZZER_DURATION_MS;
      if (!suspendSensorOutput) Serial.println("Buzzer ON due to binFull");
    }
    if (buzzerOnUntilMs != 0 && now >= buzzerOnUntilMs) {
      digitalWrite(BUZZER_PIN, LOW);
      buzzerOnUntilMs = 0;
      if (!suspendSensorOutput) Serial.println("Buzzer OFF");
    }

    if (WiFi.status() == WL_CONNECTED && client.connected()) {
      // publish ultrasonic & bin info using sensors 1 & 2
      publishUltrasonicBlock((float)d1, (float)d2, stableBinFull);

      // publish flood payload INCLUDING adc/avg/ema for remote tuning
      String floodPayload = String("{\"id\":\"") + deviceId
        + String("\",\"flooded\":") + (flooded ? "true" : "false")
        + String(",\"adc\":") + String(rawVal)
        + String(",\"avg\":") + String(avgVal);
      if (USE_EMA) floodPayload += String(",\"ema\":") + String((int)(emaVal + 0.5));
      floodPayload += String("}");
      client.beginMessage(floodTopic);
      client.print(floodPayload);
      client.endMessage();
      if (!suspendSensorOutput) Serial.println("Published flood: " + floodPayload);

      if (pendingDetection && pendingDetectionPayload.length()>0) {
        bool ok = publishDetectionPayload(pendingDetectionPayload, false);
        if (ok) {
          pendingDetection = false;
          pendingDetectionPayload = "";
        } else {
          if (!suspendSensorOutput) Serial.println("Pending detection publish failed, will retry later or on COLLECT");
        }
      }

      if (collectRequested) {
        performCollectionActions();
      }

    } else {
      if (!suspendSensorOutput) Serial.println("MQTT not connected - readings taken locally");
      if (collectRequested) {
        performCollectionActions();
        if (pendingDetection && pendingDetectionPayload.length() > 0) {
          if (!suspendSensorOutput) Serial.println("Pending detection (no MQTT): " + pendingDetectionPayload);
          lastDetectionPublishSuccess = false;
          lastDetectionPublishMs = millis();
        }
      }
    }
  } // end sensor block

  delay(10);
}
