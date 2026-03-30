/*
 * ESP32 Plant Sensor - Streaming with Display (Filtered Version)
 * WITH LAMP CONTROL VIA SHELLY PLUG
 * 
 * Features:
 * - Real-time voltage streaming via WebSocket
 * - Aggressive digital lowpass filtering (~8Hz cutoff to match Oxocard)
 * - OLED display showing IP and streaming status
 * - Button on GPIO4 to toggle Shelly smart plug (lamp)
 * 
 * Hardware: ESP32 DevKit + AD8232 + SSD1306 OLED Display + Arcade Button
 * 
 * AD8232 Connections:
 * - 3.3V → 3.3V rail
 * - GND → GND rail  
 * - Output → GPIO0 (A0) - adjust for your board
 * - SDN → 3.3V (keep active)
 * - Plant electrodes via 3.5mm jack
 * 
 * OLED Display Connections (I2C):
 * - SDA → GPIO5
 * - SCL → GPIO6
 * - VCC → 3.3V
 * - GND → GND
 * 
 * Button Connections:
 * - One terminal → GPIO4
 * - Other terminal → GND
 */

#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <HTTPClient.h>

// ===== WIFI CONFIGURATION =====
const char* ssid = "silent_signals";          // Change this!
const char* password = "39381156";          // Change this!

// ===== SHELLY PLUG CONFIGURATION =====
const char* SHELLY_IP = "192.168.1.140";      // Change to your Shelly's IP address!
bool lampState = false;                        // Track lamp on/off state

// ===== BUTTON CONFIGURATION =====
const int BUTTON_PIN = 4;                      // GPIO4 for button
unsigned long lastButtonPress = 0;
const unsigned long DEBOUNCE_DELAY = 300;      // 300ms debounce
bool lastButtonState = HIGH;                   // Button uses INPUT_PULLUP (HIGH = not pressed)

// ===== WEBSOCKET SERVER =====
WebSocketsServer webSocket = WebSocketsServer(81);  // WebSocket on port 81

// ===== OLED DISPLAY CONFIGURATION =====
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ===== SAMPLING CONFIGURATION =====
const int SAMPLE_RATE = 380;           // 380Hz internal sampling
const int OUTPUT_RATE = 100;           // 100Hz output rate (less decimation)
const int SAMPLE_INTERVAL_US = 1000000 / SAMPLE_RATE;
const int DECIMATION_FACTOR = SAMPLE_RATE / OUTPUT_RATE;  // ~4

// ===== ADC CONFIGURATION - AD8232 Optimized =====
const int ADC_PIN = A0;  // Adjust for your board. 34 for ESP32D, A0 for ESP32C
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
const unsigned long DISPLAY_UPDATE_INTERVAL = 2000;

// ═══════════════════════════════════════════════
// DIGITAL FILTER CONFIGURATION
// ═══════════════════════════════════════════════
// 
// Two-stage filtering to match Oxocard's 0.07-8.8Hz bandpass:
// 1. IIR Lowpass at ~8Hz (removes 50Hz noise and high-freq hash)
// 2. IIR Highpass at ~0.1Hz (removes DC drift)
//
// Filter coefficients calculated for 380Hz sample rate

// === STAGE 1: 20Hz Lowpass ===
// Higher cutoff to preserve touch transients while still reducing noise
// y[n] = alpha * x[n] + (1-alpha) * y[n-1]
// alpha = dt / (RC + dt), where RC = 1/(2*pi*fc)
// For fc=20Hz at 380Hz sample rate, alpha ≈ 0.27

const float LP_ALPHA = 0.25;  // Lowpass coefficient (~20Hz cutoff)
float lpFilterState = 0.0;

// === STAGE 2: 0.1Hz Highpass (DC blocking) ===
// Removes electrode drift and DC offset
// y[n] = alpha * (y[n-1] + x[n] - x[n-1])
// For fc=0.1Hz at 380Hz, alpha ≈ 0.9983

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
void setupButton();
void startWiFi();
void checkWiFiAndReconnect();
float readAndFilterSensor();
void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length);
void sendBatchData();
float adcToVoltage(int adcValue);
float applyLowpass(float input);
float applyHighpass(float input);
void checkButton();
void toggleLamp();
void setLamp(bool state);

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n╔══════════════════════════════════════════════╗");
    Serial.println("║  ESP32 Plant Sensor - Filtered + Lamp Ctrl   ║");
    Serial.println("╚══════════════════════════════════════════════╝");
    Serial.println();
    Serial.println("Features:");
    Serial.println("  • Internal sampling: 380Hz");
    Serial.println("  • Output rate: 50Hz (filtered & decimated)");
    Serial.println("  • Bandpass filter: 0.1-8Hz (matches Oxocard)");
    Serial.println("  • AD8232 bioelectric amplifier");
    Serial.println("  • Button on GPIO4 → Shelly lamp control");
    Serial.println();
    
    // Initialize Button
    setupButton();
    
    // Initialize OLED Display
    setupOLED();
    
    // Initialize ADC for AD8232
    setupADC();
    
    // Start WiFi non-blocking
    startWiFi();
    
    Serial.printf("✓ WebSocket server started on port 81\n");
    Serial.printf("✓ Internal sample rate: %d Hz\n", SAMPLE_RATE);
    Serial.printf("✓ Output sample rate: %d Hz\n", OUTPUT_RATE);
    Serial.printf("✓ Decimation factor: %d\n", DECIMATION_FACTOR);
    Serial.printf("✓ Lowpass cutoff: ~20 Hz\n");
    Serial.printf("✓ Highpass cutoff: ~0.1 Hz\n");
    Serial.printf("✓ Shelly plug IP: %s\n", SHELLY_IP);
    Serial.println();
    Serial.println("╔═══ CONNECTION INFO ═══════════════════════════╗");
    Serial.printf("║ WebSocket: ws://%s:81\n", WiFi.localIP().toString().c_str());
    Serial.printf("║ Shelly:    http://%s\n", SHELLY_IP);
    Serial.println("╚═══════════════════════════════════════════════╝");
    Serial.println();
    
    updateDisplay();
    
    lastSampleTime = micros();
    lastBatchSend = millis();
    
    // Initialize filter states with first reading
    int initialReading = analogRead(ADC_PIN);
    float initialVoltage = adcToVoltage(initialReading);
    lpFilterState = initialVoltage;
    hpPrevInput = initialVoltage;
    hpFilterState = 0.0;  // Start with zero DC offset
    
    Serial.println("Starting data streaming in 2 seconds...\n");
    delay(2000);
    Serial.println("🌱 Streaming filtered plant voltage data...\n");
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
    
    // Check button state
    checkButton();
    
    unsigned long currentTime = micros();
    
    // Sample at precise 380Hz interval
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_US) {
        lastSampleTime = currentTime;
        totalSamples++;
        
        // Read raw sensor
        int rawADC = analogRead(ADC_PIN);
        float rawVoltage = adcToVoltage(rawADC);
        lastVoltage = rawVoltage;
        
        // Apply lowpass filter (8Hz cutoff)
        float lpFiltered = applyLowpass(rawVoltage);
        
        // Apply highpass filter (0.1Hz cutoff - removes DC drift)
        float filtered = applyHighpass(lpFiltered);
        
        // Add DC offset back to center around ~800mV for display
        // (highpass removes DC, but we want to show absolute values)
        float outputVoltage = lpFiltered;  // Use lowpass only for now
        lastFilteredVoltage = outputVoltage;
        
        // Decimation: accumulate and average
        decimationAccumulator += outputVoltage;
        decimationCounter++;
        
        if (decimationCounter >= DECIMATION_FACTOR) {
            // Output one decimated sample
            float decimatedVoltage = decimationAccumulator / DECIMATION_FACTOR;
            outputSamples++;
            
            // Add to streaming buffer
            voltageBuffer[bufferIndex] = decimatedVoltage;
            bufferIndex++;
            
            // Reset decimation
            decimationAccumulator = 0.0;
            decimationCounter = 0;
            
            // When buffer is full, send
            if (bufferIndex >= BUFFER_SIZE) {
                sendBatchData();
                bufferIndex = 0;
            }
        }
        
        // Debug output every 1000 internal samples (~2.6 seconds)
        if (totalSamples % 1000 == 0) {
            Serial.printf("[Sample %lu] Raw: %.1f mV | Filtered: %.1f mV | Output: %lu\n", 
                          totalSamples, rawVoltage, lastFilteredVoltage, outputSamples);
        }
    }
    
    // Backup batch send mechanism
    if (bufferIndex > 0 && (millis() - lastBatchSend) >= BATCH_SEND_INTERVAL_MS) {
        sendBatchData();
        bufferIndex = 0;
    }
    
    // Update display periodically
    if (millis() - lastDisplayUpdate >= DISPLAY_UPDATE_INTERVAL) {
        updateDisplay();
        lastDisplayUpdate = millis();
    }
}

// ═══════════════════════════════════════════════
// BUTTON AND LAMP CONTROL
// ═══════════════════════════════════════════════

void setupButton() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    Serial.println("✓ Button configured on GPIO4 (INPUT_PULLUP)");
}

void checkButton() {
    bool currentButtonState = digitalRead(BUTTON_PIN);
    
    // Detect falling edge (button pressed - goes from HIGH to LOW)
    if (currentButtonState == LOW && lastButtonState == HIGH) {
        unsigned long currentTime = millis();
        
        // Debounce check
        if (currentTime - lastButtonPress > DEBOUNCE_DELAY) {
            lastButtonPress = currentTime;
            Serial.println("🔘 Button pressed - toggling lamp");
            toggleLamp();
        }
    }
    
    lastButtonState = currentButtonState;
}

void toggleLamp() {
    lampState = !lampState;
    setLamp(lampState);
}

void setLamp(bool state) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("⚠️  WiFi not connected - cannot control lamp");
        return;
    }
    
    HTTPClient http;
    
    // Build Shelly API URL
    String url = "http://";
    url += SHELLY_IP;
    url += "/relay/0?turn=";
    url += (state ? "on" : "off");
    
    Serial.printf("💡 Sending to Shelly: %s\n", url.c_str());
    
    http.begin(url);
    http.setTimeout(2000);  // 2 second timeout
    
    int httpCode = http.GET();
    
    if (httpCode > 0) {
        if (httpCode == HTTP_CODE_OK) {
            String payload = http.getString();
            Serial.printf("✓ Lamp turned %s\n", state ? "ON" : "OFF");
            Serial.printf("  Shelly response: %s\n", payload.c_str());
        } else {
            Serial.printf("⚠️  HTTP error code: %d\n", httpCode);
        }
    } else {
        Serial.printf("✗ HTTP request failed: %s\n", http.errorToString(httpCode).c_str());
        // Revert state on failure
        lampState = !lampState;
    }
    
    http.end();
    
    // Update display immediately to show lamp state
    updateDisplay();
}

// ═══════════════════════════════════════════════
// DIGITAL FILTERS
// ═══════════════════════════════════════════════

float applyLowpass(float input) {
    // 1st-order IIR lowpass: y = alpha*x + (1-alpha)*y_prev
    lpFilterState = LP_ALPHA * input + (1.0 - LP_ALPHA) * lpFilterState;
    return lpFilterState;
}

float applyHighpass(float input) {
    // 1st-order IIR highpass: y = alpha * (y_prev + x - x_prev)
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
    
    // For ESP32 DevKit C, you may need to specify pins:
    Wire.begin(5, 6);  // SDA=21, SCL=22. or SDA=5, SCL=6  
    
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
    display.println("Plant Sensor");
    display.println("(Filtered + Lamp)");
    display.println();
    display.println("Initializing...");
    display.display();
    delay(1000);
}

void updateDisplay() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    
    // Line 1: Title with lamp status
    display.setCursor(0, 0);
    display.print("Plant ");
    display.print(lampState ? "[LAMP ON]" : "[lamp off]");
    
    // Line 2: WiFi status and IP
    display.setCursor(0, 12);
    if (WiFi.status() == WL_CONNECTED) {
        display.print("IP:");
        display.println(WiFi.localIP().toString().c_str());
    } else {
        display.println("WiFi: Disconnected");
    }
    
    // Line 3: WebSocket clients
    display.setCursor(0, 24);
    display.print("Clients: ");
    display.println(connectedClients);
    
    // Line 4: Output sample count
    display.setCursor(0, 36);
    display.print("Out: ");
    if (outputSamples < 10000) {
        display.println(outputSamples);
    } else {
        display.print(outputSamples / 1000);
        display.println("k");
    }
    
    // Line 5: Filter info
    display.setCursor(0, 48);
    display.print("Filter: 0.1-20Hz");
    
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
            StaticJsonDocument<256> statusDoc;
            statusDoc["type"] = "status";
            statusDoc["message"] = "Connected to ESP32 Plant Sensor (Filtered + Lamp)";
            statusDoc["sampleRate"] = OUTPUT_RATE;  // Report output rate
            statusDoc["internalRate"] = SAMPLE_RATE;
            statusDoc["batchSize"] = BUFFER_SIZE;
            statusDoc["filterLow"] = "0.1Hz";
            statusDoc["filterHigh"] = "8Hz";
            statusDoc["lampState"] = lampState;
            
            String statusJson;
            serializeJson(statusDoc, statusJson);
            webSocket.sendTXT(num, statusJson);
            break;
        }
        
        case WStype_TEXT:
            Serial.printf("WebSocket: Received text from #%u: %s\n", num, payload);
            
            // Handle lamp control commands via WebSocket
            if (strcmp((char*)payload, "lamp_on") == 0) {
                setLamp(true);
            } else if (strcmp((char*)payload, "lamp_off") == 0) {
                setLamp(false);
            } else if (strcmp((char*)payload, "lamp_toggle") == 0) {
                toggleLamp();
            }
            break;
    }
}

void sendBatchData() {
    if (connectedClients == 0 || bufferIndex == 0) {
        return;
    }
    
    StaticJsonDocument<1024> doc;
    doc["type"] = "data";
    doc["timestamp"] = millis();
    doc["sampleRate"] = OUTPUT_RATE;
    doc["samples"] = bufferIndex;
    doc["lampState"] = lampState;  // Include lamp state in data packets
    
    JsonArray voltages = doc.createNestedArray("voltages");
    for (int i = 0; i < bufferIndex; i++) {
        voltages.add(voltageBuffer[i]);
    }
    
    String jsonString;
    serializeJson(doc, jsonString);
    
    webSocket.broadcastTXT(jsonString);
    packetsSent++;
    
    lastBatchSend = millis();
    
    if (packetsSent % 10 == 0) {
        Serial.printf("WebSocket: Packet #%d sent (%d samples @ %dHz)\n", 
                      packetsSent, bufferIndex, OUTPUT_RATE);
    }
}
