/*
 * ESP32-C3 Mini Plant Sensor - Bowl Exhibit with Button + Solenoid
 * 
 * Features:
 * - Real-time voltage streaming via WebSocket
 * - Digital lowpass filtering (~20Hz cutoff)
 * - OLED display showing IP and streaming status
 * - Physical button (GPIO4) to trigger solenoid strike
 * - Solenoid control via relay module (GPIO2)
 * - MODE SUPPORT: "auto" (training) vs "manual" (exhibition)
 * 
 * Hardware: ESP32-C3 Mini + AD8232 + SSD1306 OLED + 1-Channel Relay Module + Arcade Button
 * 
 * AD8232 Connections:
 * - 3.3V → 3.3V rail
 * - GND → GND rail  
 * - Output → GPIO0 (A0)
 * - SDN → 3.3V (keep active)
 * - Plant electrodes via 3.5mm jack
 * 
 * OLED Display Connections (I2C):
 * - SDA → GPIO5
 * - SCL → GPIO6
 * - VCC → 3.3V
 * - GND → GND
 * 
 * Relay Module Connections:
 * - VCC → 5V (from USB-C breakout)
 * - GND → GND (common with ESP32)
 * - IN → GPIO2
 * - COM → 5V (from USB-C breakout)
 * - NO → Solenoid (+)
 * - Solenoid (-) → GND
 * 
 * Button Connections:
 * - One terminal → GPIO4
 * - Other terminal → GND
 * 
 * WebSocket Commands:
 * - "strike"     : Trigger immediate strike
 * - "mode:auto"  : Enable automatic strikes (training mode)
 * - "mode:manual": Disable automatic strikes (exhibition mode)
 */

#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ===== WIFI CONFIGURATION =====
const char* ssid = "silent_signals";          // Change this!
const char* password = "39381156";          // Change this!

// ===== WEBSOCKET SERVER =====
WebSocketsServer webSocket = WebSocketsServer(81);  // WebSocket on port 81

// ===== OLED DISPLAY CONFIGURATION =====
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ===== BUTTON CONFIGURATION =====
const int BUTTON_PIN = 4;                      // GPIO4 for button
unsigned long lastButtonPress = 0;
const unsigned long DEBOUNCE_DELAY = 300;      // 300ms debounce
bool lastButtonState = HIGH;                   // Button uses INPUT_PULLUP (HIGH = not pressed)

// ===== RELAY/SOLENOID CONFIGURATION =====
const int RELAY_PIN = 2;                        // GPIO2 → relay IN terminal

// Random strike timing (in milliseconds) - for auto/training mode
const unsigned long MIN_STRIKE_INTERVAL_MS = 20000;   // Minimum 20 seconds
const unsigned long MAX_STRIKE_INTERVAL_MS = 40000;   // Maximum 40 seconds
const unsigned long STRIKE_DURATION_MS = 70;          // 70ms pulse

unsigned long nextStrikeTime = 0;
unsigned long strikeStartTime = 0;
bool solenoidActive = false;
int strikeCount = 0;

// ===== STRIKE MODE =====
// "auto" = automatic strikes for training
// "manual" = only button/websocket strikes for exhibition (DEFAULT for bowl exhibit)
String strikeMode = "manual";  // Default to manual for exhibition

// ===== SAMPLING CONFIGURATION =====
const int SAMPLE_RATE = 380;           // 380Hz internal sampling
const int OUTPUT_RATE = 100;           // 100Hz output rate
const int SAMPLE_INTERVAL_US = 1000000 / SAMPLE_RATE;
const int DECIMATION_FACTOR = SAMPLE_RATE / OUTPUT_RATE;  // ~4

// ===== ADC CONFIGURATION - AD8232 on ESP32-C3 =====
const int ADC_PIN = A0;  // GPIO0 on ESP32-C3
const int ADC_RESOLUTION = 12;
const float ADC_VOLTAGE_REF = 3.3;

// ===== STREAMING BUFFER =====
const int BUFFER_SIZE = 50;  // Send 50 samples at a time (500ms at 100Hz)
float voltageBuffer[BUFFER_SIZE];
int bufferIndex = 0;

// ===== TIMING =====
unsigned long lastSampleTime = 0;
unsigned long lastBatchSend = 0;
const int BATCH_SEND_INTERVAL_MS = 500;

// ===== WEBSOCKET CLIENT TRACKING =====
int connectedClients = 0;
int packetsSent = 0;

// ===== STATUS VARIABLES =====
unsigned long totalSamples = 0;
unsigned long outputSamples = 0;
float lastVoltage = 0.0;
float lastFilteredVoltage = 0.0;
String wifi_status = "Disconnected";
String websocket_status = "Stopped";

// ===== WIFI RECONNECT TIMING =====
unsigned long lastWifiCheck = 0;
const unsigned long WIFI_CHECK_INTERVAL = 5000;
bool webSocketStarted = false;

// ===== DISPLAY UPDATE TIMING =====
unsigned long lastDisplayUpdate = 0;
const unsigned long DISPLAY_UPDATE_INTERVAL = 1000;  // Update every second

// ═══════════════════════════════════════════════
// DIGITAL FILTER CONFIGURATION
// ═══════════════════════════════════════════════

const float LP_ALPHA = 0.25;  // Lowpass coefficient (~20Hz cutoff)
float lpFilterState = 0.0;

const float HP_ALPHA = 0.998;  // Highpass coefficient (~0.1Hz cutoff)
float hpFilterState = 0.0;
float hpPrevInput = 0.0;

// === Decimation counter ===
int decimationCounter = 0;
float decimationAccumulator = 0.0;

// ===== FUNCTION DECLARATIONS =====
void setupOLED();
void updateDisplay();
void setupADC();
void setupRelay();
void setupButton();
void checkButton();
void handleSolenoid();
void triggerStrike();
void scheduleNextStrike();
void sendStrikeEvent();
void sendModeEvent();
void startWiFi();
void checkWiFiAndReconnect();
void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length);
void sendBatchData();
float adcToVoltage(int adcValue);
float applyLowpass(float input);
float applyHighpass(float input);

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n╔══════════════════════════════════════════════════╗");
    Serial.println("║  ESP32-C3 Plant Sensor - Bowl Exhibit            ║");
    Serial.println("║  Button + Solenoid Control                       ║");
    Serial.println("╚══════════════════════════════════════════════════╝");
    Serial.println();
    Serial.println("Features:");
    Serial.println("  • Internal sampling: 380Hz");
    Serial.println("  • Output rate: 100Hz (filtered & decimated)");
    Serial.println("  • Bandpass filter: 0.1-20Hz");
    Serial.println("  • AD8232 bioelectric amplifier");
    Serial.println("  • Button on GPIO4 → trigger bowl strike");
    Serial.println("  • Relay on GPIO2 → solenoid control");
    Serial.println();
    Serial.println("Commands:");
    Serial.println("  • Button press  - Trigger strike");
    Serial.println("  • 'strike'      - Trigger via WebSocket");
    Serial.println("  • 'mode:auto'   - Auto strikes (training)");
    Serial.println("  • 'mode:manual' - Manual only (exhibition)");
    Serial.println();
    
    // Initialize random seed from analog noise
    randomSeed(analogRead(0) + micros());
    
    // Initialize Button
    setupButton();
    
    // Initialize OLED Display
    setupOLED();
    
    // Initialize ADC for AD8232
    setupADC();
    
    // Initialize Relay
    setupRelay();
    
    // Start WiFi non-blocking
    startWiFi();
    
    // Re-initialize relay pin — GPIO2 is a strapping pin on ESP32-C3
    // and can lose its state during WiFi init
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, LOW);
    
    Serial.printf("✓ WebSocket server started on port 81\n");
    Serial.printf("✓ Internal sample rate: %d Hz\n", SAMPLE_RATE);
    Serial.printf("✓ Output sample rate: %d Hz\n", OUTPUT_RATE);
    Serial.printf("✓ Default mode: %s\n", strikeMode.c_str());
    Serial.println();
    Serial.println("╔═══ CONNECTION INFO ═══════════════════════════╗");
    Serial.printf("║ WebSocket: ws://%s:81\n", WiFi.localIP().toString().c_str());
    Serial.println("╚═══════════════════════════════════════════════╝");
    Serial.println();
    
    // Schedule first strike only if in auto mode
    if (strikeMode == "auto") {
        scheduleNextStrike();
    }
    
    updateDisplay();
    
    lastSampleTime = micros();
    lastBatchSend = millis();
    
    // Initialize filter states with first reading
    int initialReading = analogRead(ADC_PIN);
    float initialVoltage = adcToVoltage(initialReading);
    lpFilterState = initialVoltage;
    hpPrevInput = initialVoltage;
    hpFilterState = 0.0;
    
    Serial.println("Starting data streaming in 2 seconds...\n");
    delay(2000);
    Serial.println("🌱 Streaming filtered plant voltage data...\n");
    Serial.println("🔘 Press button to strike the bowl!\n");
}

void loop() {
    // WiFi watchdog
    if (millis() - lastWifiCheck >= WIFI_CHECK_INTERVAL) {
        lastWifiCheck = millis();
        checkWiFiAndReconnect();
    }
    // WebSocket: start once WiFi is up
    if (WiFi.status() == WL_CONNECTED) {
        if (!webSocketStarted) {
            webSocket.begin();
            webSocket.onEvent(webSocketEvent);
            webSocketStarted = true;
            websocket_status = "Ready";
            Serial.printf("✓ WebSocket started on ws://%s:81\n", WiFi.localIP().toString().c_str());
        }
        webSocket.loop();
    }
    
    unsigned long currentTime = micros();
    unsigned long currentMillis = millis();
    
    // Check button state
    checkButton();
    
    // Handle solenoid timing
    handleSolenoid();
    
    // Sample at precise 380Hz interval
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_US) {
        lastSampleTime = currentTime;
        totalSamples++;
        
        // Read raw sensor
        int rawADC = analogRead(ADC_PIN);
        float rawVoltage = adcToVoltage(rawADC);
        lastVoltage = rawVoltage;
        
        // Apply lowpass filter
        float lpFiltered = applyLowpass(rawVoltage);
        
        // Apply highpass filter (removes DC drift)
        float filtered = applyHighpass(lpFiltered);
        
        float outputVoltage = lpFiltered;
        lastFilteredVoltage = outputVoltage;
        
        // Decimation: accumulate and average
        decimationAccumulator += outputVoltage;
        decimationCounter++;
        
        if (decimationCounter >= DECIMATION_FACTOR) {
            float decimatedVoltage = decimationAccumulator / DECIMATION_FACTOR;
            outputSamples++;
            
            voltageBuffer[bufferIndex] = decimatedVoltage;
            bufferIndex++;
            
            decimationAccumulator = 0.0;
            decimationCounter = 0;
            
            if (bufferIndex >= BUFFER_SIZE) {
                sendBatchData();
                bufferIndex = 0;
            }
        }
        
        // Debug output every 1000 internal samples (~2.6 seconds)
        if (totalSamples % 1000 == 0) {
            Serial.printf("[%lu] Raw: %.1fmV | Filtered: %.1fmV | Mode: %s | Strikes: %d\n",
                         totalSamples, lastVoltage, lastFilteredVoltage, strikeMode.c_str(), strikeCount);
        }
    }
    
    // Update display periodically
    if (currentMillis - lastDisplayUpdate >= DISPLAY_UPDATE_INTERVAL) {
        lastDisplayUpdate = currentMillis;
        updateDisplay();
    }
}

// ═══════════════════════════════════════════════
// BUTTON CONTROL
// ═══════════════════════════════════════════════

void setupButton() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    Serial.printf("✓ Button configured on GPIO%d (INPUT_PULLUP)\n", BUTTON_PIN);
}

void checkButton() {
    bool currentButtonState = digitalRead(BUTTON_PIN);
    
    // Detect falling edge (button pressed - goes from HIGH to LOW)
    if (currentButtonState == LOW && lastButtonState == HIGH) {
        unsigned long currentTime = millis();
        
        // Debounce check
        if (currentTime - lastButtonPress > DEBOUNCE_DELAY) {
            lastButtonPress = currentTime;
            Serial.println("🔘 Button pressed - triggering bowl strike!");
            triggerStrike();
        }
    }
    
    lastButtonState = currentButtonState;
}

// ═══════════════════════════════════════════════
// RELAY/SOLENOID CONTROL
// ═══════════════════════════════════════════════

void setupRelay() {
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, LOW);  // Relay off initially
    Serial.printf("✓ Relay configured on GPIO%d\n", RELAY_PIN);
}

void handleSolenoid() {
    unsigned long currentMillis = millis();
    
    // Turn off solenoid after pulse duration
    if (solenoidActive && (currentMillis - strikeStartTime >= STRIKE_DURATION_MS)) {
        digitalWrite(RELAY_PIN, LOW);
        solenoidActive = false;
        Serial.println("   Solenoid OFF");
        
        // Schedule next strike (only in auto mode)
        if (strikeMode == "auto") {
            scheduleNextStrike();
        }
    }
    
    // Check if it's time for automatic strike (only in auto mode)
    if (strikeMode == "auto" && !solenoidActive && currentMillis >= nextStrikeTime) {
        triggerStrike();
    }
}

void scheduleNextStrike() {
    unsigned long interval = random(MIN_STRIKE_INTERVAL_MS, MAX_STRIKE_INTERVAL_MS + 1);
    nextStrikeTime = millis() + interval;
    
    Serial.printf("📅 Next automatic strike scheduled in %lu seconds\n", interval / 1000);
}

void triggerStrike() {
    // Prevent retriggering while solenoid is active
    if (solenoidActive) {
        Serial.println("⚠️  Strike ignored - solenoid already active");
        return;
    }
    
    strikeCount++;
    strikeStartTime = millis();
    solenoidActive = true;
    
    digitalWrite(RELAY_PIN, HIGH);
    
    Serial.println();
    Serial.println("══════════════════════════════════════");
    Serial.printf("🔔 BOWL STRIKE #%d\n", strikeCount);
    Serial.printf("   Timestamp: %lu ms\n", strikeStartTime);
    Serial.printf("   Sample count: %lu\n", outputSamples);
    Serial.printf("   Mode: %s\n", strikeMode.c_str());
    Serial.println("══════════════════════════════════════\n");
    
    // Send event via WebSocket for data synchronization
    sendStrikeEvent();
    
    // Update display immediately
    updateDisplay();
}

void sendStrikeEvent() {
    if (connectedClients == 0) {
        return;
    }
    
    StaticJsonDocument<256> doc;
    doc["type"] = "strike";
    doc["timestamp"] = millis();
    doc["strikeNumber"] = strikeCount;
    doc["sampleCount"] = outputSamples;
    doc["mode"] = strikeMode;
    
    String jsonString;
    serializeJson(doc, jsonString);
    
    webSocket.broadcastTXT(jsonString);
    
    Serial.printf("   Strike event sent to %d client(s)\n", connectedClients);
}

void sendModeEvent() {
    if (connectedClients == 0) {
        return;
    }
    
    StaticJsonDocument<128> doc;
    doc["type"] = "mode";
    doc["mode"] = strikeMode;
    doc["timestamp"] = millis();
    
    String jsonString;
    serializeJson(doc, jsonString);
    
    webSocket.broadcastTXT(jsonString);
    
    Serial.printf("   Mode event sent: %s\n", strikeMode.c_str());
}

// ═══════════════════════════════════════════════
// DIGITAL FILTERS
// ═══════════════════════════════════════════════

float applyLowpass(float input) {
    lpFilterState = LP_ALPHA * input + (1.0 - LP_ALPHA) * lpFilterState;
    return lpFilterState;
}

float applyHighpass(float input) {
    float output = HP_ALPHA * (hpFilterState + input - hpPrevInput);
    hpFilterState = output;
    hpPrevInput = input;
    return output;
}

// ═══════════════════════════════════════════════
// DISPLAY FUNCTIONS
// ═══════════════════════════════════════════════

void setupOLED() {
    Serial.print("Initializing OLED display... ");
    
    Wire.begin(5, 6);  // SDA=5, SCL=6 for ESP32-C3 Mini
    
    if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        Serial.println("✗ FAILED");
        Serial.println("⚠️  Display not found - continuing without display");
        return;
    }
    
    Serial.println("✓");
    
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("Bowl Exhibit");
    display.println("Button + Solenoid");
    display.println();
    display.println("Initializing...");
    display.display();
    delay(1000);
}

void updateDisplay() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    
    // Line 1: Title + Mode
    display.setCursor(0, 0);
    if (strikeMode == "auto") {
        display.println("Bowl [AUTO]");
    } else {
        display.println("Bowl [MANUAL]");
    }
    
    // Line 2: WiFi status and IP
    display.setCursor(0, 12);
    if (WiFi.status() == WL_CONNECTED) {
        display.print("IP:");
        display.println(WiFi.localIP().toString().c_str());
    } else {
        display.print("WiFi: ");
        display.println(wifi_status.c_str());
    }
    
    // Line 3: WebSocket clients + strike count
    display.setCursor(0, 24);
    display.print("Clients:");
    display.print(connectedClients);
    display.print(" Hits:");
    display.println(strikeCount);
    
    // Line 4: Output sample count
    display.setCursor(0, 36);
    display.print("Samples: ");
    if (outputSamples < 10000) {
        display.println(outputSamples);
    } else {
        display.print(outputSamples / 1000);
        display.println("k");
    }
    
    // Line 5: Next strike countdown (only in auto mode) or instruction
    display.setCursor(0, 48);
    if (strikeMode == "auto") {
        unsigned long currentMillis = millis();
        if (nextStrikeTime > currentMillis) {
            unsigned long timeToNext = (nextStrikeTime - currentMillis) / 1000;
            display.print("Next hit: ");
            display.print(timeToNext);
            display.println("s");
        } else {
            display.println("Striking...");
        }
    } else {
        display.println("Press button!");
    }
    
    // Line 6: Current filtered voltage
    display.setCursor(0, 56);
    display.print("V: ");
    display.print(lastFilteredVoltage, 1);
    display.println(" mV");
    
    display.display();
}

// ═══════════════════════════════════════════════
// SETUP FUNCTIONS
// ═══════════════════════════════════════════════

void setupADC() {
    analogReadResolution(ADC_RESOLUTION);
    analogSetAttenuation(ADC_11db);
    
    Serial.println("✓ ADC configured for AD8232");
    Serial.printf("  Pin: GPIO%d (A0)\n", ADC_PIN);
    Serial.printf("  Resolution: %d bits\n", ADC_RESOLUTION);
    Serial.printf("  Reference: %.1f V\n", ADC_VOLTAGE_REF);
}

void startWiFi() {
    Serial.printf("WiFi: connecting to '%s' (non-blocking)...\n", ssid);
    wifi_status = "Connecting";
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
}

void checkWiFiAndReconnect() {
    if (WiFi.status() == WL_CONNECTED) {
        if (wifi_status != "Connected") {
            wifi_status = "Connected";
            Serial.printf("✓ WiFi connected — IP: %s  Signal: %d dBm\n",
                          WiFi.localIP().toString().c_str(), WiFi.RSSI());
        }
        return;
    }
    wifi_status = "Reconnecting";
    webSocketStarted = false;
    websocket_status = "Waiting";
    Serial.printf("WiFi lost — reconnecting to '%s'...\n", ssid);
    WiFi.disconnect();
    delay(100);
    WiFi.begin(ssid, password);
}

float adcToVoltage(int adcValue) {
    float voltage = (adcValue / 4095.0) * ADC_VOLTAGE_REF * 1000.0;
    return voltage;
}

// ═══════════════════════════════════════════════
// WEBSOCKET COMMUNICATION
// ═══════════════════════════════════════════════

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
    switch(type) {
        case WStype_DISCONNECTED:
            connectedClients--;
            websocket_status = String(connectedClients) + " client(s)";
            Serial.printf("WebSocket: Client #%u disconnected (total: %d)\n", num, connectedClients);
            break;
            
        case WStype_CONNECTED: {
            IPAddress ip = webSocket.remoteIP(num);
            connectedClients++;
            websocket_status = String(connectedClients) + " client(s)";
            Serial.printf("WebSocket: Client #%u connected from %s (total: %d)\n", 
                          num, ip.toString().c_str(), connectedClients);
            
            // Send initial status message
            StaticJsonDocument<512> statusDoc;
            statusDoc["type"] = "status";
            statusDoc["message"] = "Connected to ESP32-C3 Bowl Exhibit";
            statusDoc["sampleRate"] = OUTPUT_RATE;
            statusDoc["internalRate"] = SAMPLE_RATE;
            statusDoc["batchSize"] = BUFFER_SIZE;
            statusDoc["filterLow"] = "0.1Hz";
            statusDoc["filterHigh"] = "20Hz";
            statusDoc["strikeMode"] = strikeMode;
            statusDoc["strikeMinInterval"] = MIN_STRIKE_INTERVAL_MS / 1000;
            statusDoc["strikeMaxInterval"] = MAX_STRIKE_INTERVAL_MS / 1000;
            statusDoc["strikeCount"] = strikeCount;
            
            String statusJson;
            serializeJson(statusDoc, statusJson);
            webSocket.sendTXT(num, statusJson);
            break;
        }
        
        case WStype_TEXT: {
            String message = String((char*)payload);
            Serial.printf("WebSocket: Received from #%u: %s\n", num, payload);
            
            // Manual strike trigger
            if (message == "strike") {
                Serial.println("🔔 Strike requested via WebSocket");
                triggerStrike();
            }
            // Mode switching
            else if (message == "mode:auto") {
                strikeMode = "auto";
                scheduleNextStrike();  // Start auto-strike timer
                Serial.println("📋 Mode changed to AUTO (training)");
                sendModeEvent();
                updateDisplay();
            }
            else if (message == "mode:manual") {
                strikeMode = "manual";
                Serial.println("📋 Mode changed to MANUAL (exhibition)");
                sendModeEvent();
                updateDisplay();
            }
            break;
        }
    }
}

void sendBatchData() {
    if (connectedClients == 0 || bufferIndex == 0) {
        return;
    }
    
    StaticJsonDocument<1536> doc;
    doc["type"] = "data";
    doc["timestamp"] = millis();
    doc["sampleRate"] = OUTPUT_RATE;
    doc["samples"] = bufferIndex;
    doc["strikeCount"] = strikeCount;
    
    JsonArray voltages = doc.createNestedArray("voltages");
    for (int i = 0; i < bufferIndex; i++) {
        voltages.add(voltageBuffer[i]);
    }
    
    String jsonString;
    serializeJson(doc, jsonString);
    
    webSocket.broadcastTXT(jsonString);
    packetsSent++;
    
    lastBatchSend = millis();
    
    if (packetsSent % 20 == 0) {
        Serial.printf("WebSocket: Packet #%d sent (%d samples @ %dHz)\n", 
                      packetsSent, bufferIndex, OUTPUT_RATE);
    }
}
