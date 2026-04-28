#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/i2c.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "tmf8829_image.h"
#include "esp_vfs_dev.h"
#include "esp_vfs_usb_serial_jtag.h"

static const char *TAG = "TMF8829_INIT";


#define DATA_BUFFER_SIZE 256

// ─── Wiring & I2C Settings ────────────────────────────────────────────────────
#define I2C_MASTER_SCL_IO           9
#define I2C_MASTER_SDA_IO           8
#define I2C_MASTER_NUM              0
#define I2C_MASTER_FREQ_HZ          400000
#define I2C_MASTER_TIMEOUT_MS       100

#define TMF8829_ADDR                0x41
#define TMF8829_EN_PIN              4

// ─── Registers ────────────────────────────────────────────────────────────────
#define REG_APP_ID                  0x00
#define REG_MAJOR                   0x01
#define REG_MINOR                   0x02
#define REG_CMD_STAT                0x08
#define REG_SERIAL_0                0x1C // 1C, 1D, 1E, 1F
#define REG_ENABLE                  0xF8
#define REG_FIFO                    0xFF

// ─── Bootloader Commands ──────────────────────────────────────────────────────
// Used from Datasheet Table 5 (Converted to Hexadecimal correctly)
#define BL_CMD_START_RAM_APP        0x16 // 22 Decimal
#define BL_CMD_SPI_OFF              0x20 // 32 Decimal
#define BL_CMD_W_FIFO_BOTH          0x45 // 69 Decimal

// Application commands — change CMD_LOAD_CFG_* here to switch modes
#define CMD_LOAD_CFG_8X8                 0x40
#define CMD_LOAD_CFG_8x8_HIGH_ACCURACY   0x42
#define CMD_LOAD_CFG_16X16               0x43
#define CMD_LOAD_CFG_16x16_HIGH_ACCURACY 0x44
#define CMD_LOAD_CFG_32X32               0x45
#define CMD_LOAD_CFG_32x32_HIGH_ACCURACY 0x46
#define CMD_LOAD_CFG_48x32               0x47
#define CMD_LOAD_CFG_48x32_HIGH_ACCURACY 0x48

// MEASURE
#define CMD_MEASURE                0x10
#define CMD_WRITE_PAGE_AND_MEASURE 0x14
#define REG_INT_STATUS             0xE1
#define INT_BIT_RESULT_FRAME       0x01

/* result page addresses and defines */
#define TMF8829_PRE_HEADER_SIZE      5
#define TMF8829_FRAME_HEADER_SIZE   16
#define TMF8829_FRAME_FOOTER_SIZE   12
#define TMF8829_FRAME_EOF_SIZE       2

#define TMF8829_FRAME_HEADER_OFFSET  4        /**< the first bytes are not part of the payload value in the frame */
#define TMF8829_FRAME_EOF            0xE0F7   /**< End of Frame Marker */

#define TMF8829_CFG_RESULT_FORMAT_NR_PEAKS_MASK         0x07
#define TMF8829_CFG_RESULT_FORMAT_SIGNAL_STRENGTH_MASK  0x08
#define TMF8829_CFG_RESULT_FORMAT_NOISE_STRENGTH_MASK   0x10
#define TMF8829_CFG_RESULT_FORMAT_XTALK_MASK            0x20

#define FIFOSTATUS 0xfa


typedef struct __attribute__((packed)) { // this is to avoid 1 extra byte of padding 
    uint16_t distance_mm;
    uint8_t confidence;
} tmf8829_pixel_t;

typedef struct {
    uint16_t        num_pixels;
    tmf8829_pixel_t pixels[1536]; 
} tmf8829_frame_t;

static tmf8829_frame_t out_frame; // Global frame to hold pixel data after reading from FIFO


static const uint8_t mode = CMD_LOAD_CFG_16x16_HIGH_ACCURACY; // Change this to switch modes


// ─── Helpers ──────────────────────────────────────────────────────────────────
static esp_err_t i2c_read_reg(uint8_t reg, uint8_t *data, size_t len) {
    return i2c_master_write_read_device(I2C_MASTER_NUM, TMF8829_ADDR, &reg, 1, data, len, pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
}

static esp_err_t i2c_write_reg(uint8_t reg, uint8_t data) {
    uint8_t buf[2] = {reg, data};
    return i2c_master_write_to_device(I2C_MASTER_NUM, TMF8829_ADDR, buf, 2, pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
}

static esp_err_t i2c_write_reg_buf(uint8_t reg, const uint8_t *data, size_t len) {
    uint8_t *buf = malloc(len + 1);
    buf[0] = reg;
    memcpy(buf + 1, data, len);
    esp_err_t ret = i2c_master_write_to_device(I2C_MASTER_NUM, TMF8829_ADDR, buf, len + 1, pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
    free(buf);
    return ret;
}

static esp_err_t wait_cmd_stat(uint8_t expected) {
    uint8_t stat;
    for (int i = 0; i < 50; i++) {
        vTaskDelay(pdMS_TO_TICKS(2));
        if (i2c_read_reg(REG_CMD_STAT, &stat, 1) == ESP_OK) {
            if (stat == expected) return ESP_OK;
        }
    }
    ESP_LOGE(TAG, "Timeout waiting for cmd_stat == 0x%02X", expected);
    return ESP_ERR_TIMEOUT;
}

static bool wait_interrupt(uint32_t timeout_ms)
{
    // Calculate exact deadline in microseconds
    int64_t deadline = esp_timer_get_time() + (int64_t)timeout_ms * 1000;
    uint8_t status = 0;

    while (esp_timer_get_time() < deadline) {
        // Read the INT_STATUS register
        if (i2c_read_reg(REG_INT_STATUS, &status, 1) == ESP_OK) {
            
            // If the result frame interrupt fired
            if (status & INT_BIT_RESULT_FRAME) {
                
                // Clear it by writing the same bit back
                i2c_write_reg(REG_INT_STATUS, INT_BIT_RESULT_FRAME); 
                return true;
            }
        }
        
        // Yield 1ms to FreeRTOS to prevent watchdog crashes
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    
    return false; // Timeout reached
}


static esp_err_t tmf8829_upload_firmware(const uint8_t *fw_data, size_t fw_size) {
    ESP_LOGI(TAG, "Uploading firmware (%d bytes)...", fw_size);

    // 6. Setup FIFO Upload command 
    // Uses auto-increment to write starting from 0x08 (CMD_STAT) through 0x0F (WORD_SIZE MSB)
    uint32_t word_size = tmf8829_image_length / 4;
    uint8_t setup_cmd[8] = {
        BL_CMD_W_FIFO_BOTH, 
        0x06,                                // Payload size
        tmf8829_image_start & 0xFF,          // LSB of Address
        (tmf8829_image_start >> 8) & 0xFF,
        (tmf8829_image_start >> 16) & 0xFF,
        (tmf8829_image_start >> 24) & 0xFF,  // MSB of Address
        word_size & 0xFF,                    // LSB of Word Size
        (word_size >> 8) & 0xFF              // MSB of Word Size
    };
    i2c_write_reg_buf(REG_CMD_STAT, setup_cmd, 8);
    if (wait_cmd_stat(0) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to setup FIFO upload");
        return ESP_FAIL;
    }

    // 7. Write firmware in chunks to the FIFO address (0xFF)
    ESP_LOGI(TAG, "Downloading firmware patch to RAM...");
    int offset = 0;
    while(offset < tmf8829_image_length) {
        int chunk_size = tmf8829_image_length - offset;
        if (chunk_size > 128) chunk_size = 128; // I2C hardware buffer limit safe-zone
        i2c_write_reg_buf(REG_FIFO, tmf8829_image + offset, chunk_size);
        offset += chunk_size;
    }

    // 8. Start the RAM application
    i2c_write_reg(REG_CMD_STAT, BL_CMD_START_RAM_APP);
    if (wait_cmd_stat(0) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start RAM app");
        return ESP_FAIL;
    }

    return ESP_OK;
}

static uint8_t tmf8829GetPixelSize(uint8_t layout)
{
  uint8_t size = 0;
  uint8_t numPeak = layout & TMF8829_CFG_RESULT_FORMAT_NR_PEAKS_MASK;
  uint8_t useSignal = ( (layout & TMF8829_CFG_RESULT_FORMAT_SIGNAL_STRENGTH_MASK) > 0) ? 1 : 0;
  uint8_t useNoise =  ( (layout & TMF8829_CFG_RESULT_FORMAT_NOISE_STRENGTH_MASK) > 0) ? 1 : 0;
  uint8_t useXtalk =  ( (layout & TMF8829_CFG_RESULT_FORMAT_XTALK_MASK) > 0) ? 1 : 0;

  size = ( numPeak * ( 3 + 2 * useSignal) ) + ( 2 * useNoise ) + ( 2 * useXtalk );

  return size;
}



static esp_err_t read_frame() { 

    bool read = wait_interrupt(100); 
    if(!read) {
        // ESP_LOGD(TAG, "Timeout waiting for result frame interrupt"); // Kept as debug so it doesn't spam
        return ESP_ERR_TIMEOUT;
    }

    uint8_t pixelSize = 0;
    uint8_t header_data[TMF8829_PRE_HEADER_SIZE + TMF8829_FRAME_HEADER_SIZE];
    
    // 1. Read the 21 bytes of Pre-Header and Frame Header
    esp_err_t err = i2c_read_reg(FIFOSTATUS, header_data, sizeof(header_data));
    if (err != ESP_OK) return ESP_FAIL;

    // 2. Check if this is a Result Frame (Frame ID 0x10)
    if ((header_data[TMF8829_PRE_HEADER_SIZE] & 0xF0) != 0x10) {
        return ESP_FAIL;
    }

    // 3. Extract the Header fields
    const uint8_t *fh = &header_data[TMF8829_PRE_HEADER_SIZE];
    uint8_t  layout_raw = fh[1];                            
    uint8_t  layout     = layout_raw & 0x0F; // Mask out the sub-frame flag to get the real pixel layout
    uint16_t payload    = fh[2] | (fh[3] << 8);             
    
    pixelSize = tmf8829GetPixelSize(layout);

    // ─── STITCHING LOGIC BASED ON MODE ────────────────────────────────────────
    
    // Determine Width based on your global `mode` variable
    uint16_t width = 8;
    if (mode == CMD_LOAD_CFG_16X16 || mode == CMD_LOAD_CFG_16x16_HIGH_ACCURACY) {
        width = 16;
    } else if (mode == CMD_LOAD_CFG_32X32 || mode == CMD_LOAD_CFG_32x32_HIGH_ACCURACY) {
        width = 32;
    }

    // Check if this is the "Odd Rows" sub-frame (indicated by upper bits of layout)
    bool is_odd_rows = (layout_raw & 0xF0) != 0;

    // If this is the FIRST half of a frame (or standard 8x8/16x16), clear our global frame
    if (!is_odd_rows) {
        out_frame.num_pixels = 0;
    }

    // ──────────────────────────────────────────────────────────────────────────

    uint16_t sizeToRead = payload - 12;
    uint8_t dataBuffer[DATA_BUFFER_SIZE];
    uint16_t eofMarker = 0;
    uint16_t sub_pixel_idx = 0; // Tracks the pixel position inside THIS specific chunk

    // 4. Read the remaining pixel data
    while (sizeToRead > 0) {
        uint16_t rxSize = (sizeToRead > DATA_BUFFER_SIZE) ? ((DATA_BUFFER_SIZE / pixelSize) * pixelSize) : sizeToRead;
        sizeToRead -= rxSize;
        
        if (i2c_read_reg(REG_FIFO, dataBuffer, rxSize) != ESP_OK) return ESP_FAIL;

        uint16_t bytesToParse = rxSize;
        if (sizeToRead == 0 && bytesToParse >= 12) {
            bytesToParse -= 12; // Exclude footer
            eofMarker = dataBuffer[rxSize - 2] | (dataBuffer[rxSize - 1] << 8);
        }

        // 5. Parse pixel data
        for (uint16_t i = 0; i < bytesToParse; i += pixelSize) {
            uint16_t raw_distance = 0;
            uint8_t confidence = 0;
            
            // Handle different layouts
            if (layout == 0x01) { 
                raw_distance = dataBuffer[i + 0] | (dataBuffer[i + 1] << 8);
                confidence = dataBuffer[i + 2];
            } else if (layout == 0x11) { 
                raw_distance = dataBuffer[i + 2] | (dataBuffer[i + 3] << 8);
                confidence = dataBuffer[i + 4];
            }
            
            // ─── STITCHING MATH ──────────────────────────────────────────────
            // Figure out the X/Y row and column for this pixel
            uint16_t row = sub_pixel_idx / width;
            uint16_t col = sub_pixel_idx % width;

            // If we are in 32x32 mode, space the rows out!
            if (width == 32) {
                row = (row * 2) + (is_odd_rows ? 1 : 0);
            }

            // Convert back to a flat array index
            uint16_t abs_idx = (row * width) + col;

            // Save to our global struct
            if (abs_idx < 1536) {
                out_frame.pixels[abs_idx].distance_mm = raw_distance / 4;
                out_frame.pixels[abs_idx].confidence = confidence;
                out_frame.num_pixels++;
            }
            sub_pixel_idx++;
        }
    }

    if (eofMarker == TMF8829_FRAME_EOF) return ESP_OK;
    return ESP_FAIL;
}


void send_frame_binary(tmf8829_frame_t *frame) {
    // 1. Send a Sync Word (0xAA 0x55) so the PC knows a frame is starting
    const uint8_t sync_word[2] = {0xAA, 0x55};
    fwrite(sync_word, 1, 2, stdout);

    // 2. Send the number of pixels (2 bytes, Little Endian)
    fwrite(&frame->num_pixels, sizeof(uint16_t), 1, stdout);

    // 3. Dump the entire pixel array in one massive, fast DMA transfer
    // Size = num_pixels * 3 bytes (since we packed the struct!)
    fwrite(frame->pixels, sizeof(tmf8829_pixel_t), frame->num_pixels, stdout);

    // 4. Send an End Word (0xEF 0xBE) to close the frame
    const uint8_t end_word[2] = {0xEF, 0xBE};
    fwrite(end_word, 1, 2, stdout);

    // 5. Force the ESP32 to push the data out of the UART buffer immediately
    fflush(stdout); 
}

// ─── Main Application ─────────────────────────────────────────────────────────
void app_main(void) {
    esp_log_level_set("*", ESP_LOG_NONE); // Disable all logging for cleaner binary output
    esp_vfs_dev_usb_serial_jtag_set_tx_line_endings(ESP_LINE_ENDINGS_LF);
    //esp_vfs_dev_uart_port_set_tx_line_endings(0, ESP_LINE_ENDINGS_LF);
    vTaskDelay(pdMS_TO_TICKS(2000));
    // 1. Initialize I2C Master
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };
    i2c_param_config(I2C_MASTER_NUM, &conf);
    i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);

    // 2. Hardware Reset / Power Up
    gpio_reset_pin(TMF8829_EN_PIN);
    gpio_set_direction(TMF8829_EN_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(TMF8829_EN_PIN, 0);
    vTaskDelay(pdMS_TO_TICKS(10));
    gpio_set_level(TMF8829_EN_PIN, 1);
    vTaskDelay(pdMS_TO_TICKS(10)); // Allow time to bootup

    // 3. Wake Up sequence
    i2c_write_reg(REG_ENABLE, 0x04); // pon=1, powerup_select=0
    vTaskDelay(pdMS_TO_TICKS(5));

    uint8_t enable_reg = 0;
    bool cpu_ready = false;
    for(int i = 0; i < 10; i++) {
        i2c_read_reg(REG_ENABLE, &enable_reg, 1);
        if (enable_reg & 0x80) { cpu_ready = true; break; } // bit 7 indicates cpu_ready == 1
        vTaskDelay(pdMS_TO_TICKS(2));
    }
    if (!cpu_ready) {
        ESP_LOGE(TAG, "CPU not ready. ENABLE register: 0x%02X", enable_reg);
        return;
    }

    // 4. Check bootloader ID
    uint8_t app_id;
    i2c_read_reg(REG_APP_ID, &app_id, 1);
    if (app_id != 0x80) {
        ESP_LOGE(TAG, "Not in bootloader mode, APP_ID=0x%02X", app_id);
        return;
    }
    ESP_LOGI(TAG, "In bootloader mode. Proceeding to flash...");

    // 5. Disable SPI interface to isolate I2C communications
    i2c_write_reg(REG_CMD_STAT, BL_CMD_SPI_OFF);
    if (wait_cmd_stat(0) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to disable SPI");
        return;
    }

    tmf8829_upload_firmware(tmf8829_image, tmf8829_image_length);


    // 9. Read App ID, Version, and Serial Number
    vTaskDelay(pdMS_TO_TICKS(10)); // Give app time to map logic
    i2c_read_reg(REG_APP_ID, &app_id, 1);
    
    if (app_id == 0x01) { // 0x01 is the TMF8829 Rom Application ID
        uint8_t major, minor;
        i2c_read_reg(REG_MAJOR, &major, 1);
        i2c_read_reg(REG_MINOR, &minor, 1);
        ESP_LOGI(TAG, "Firmware loaded successfully! App ID: 0x%02X, Version: %d.%d", app_id, major, minor);

        // Fetch Serial number which is 4 bytes starting at REG_SERIAL_0 (0x1C)
        uint8_t serial[4];
        i2c_read_reg(REG_SERIAL_0, serial, 4);
        ESP_LOGI(TAG, "Device Serial Number: 0x%02X%02X%02X%02X", serial[3], serial[2], serial[1], serial[0]);
    } else {
        ESP_LOGE(TAG, "Failed to load firmware, returned APP_ID=0x%02X", app_id);
    }

    i2c_write_reg(REG_CMD_STAT, mode);
    if(wait_cmd_stat(0) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to load mode");
        return;
    }

    vTaskDelay(pdMS_TO_TICKS(10)); // Short delay before starting measurement



    //We can now start measuring

    i2c_write_reg(REG_CMD_STAT, CMD_WRITE_PAGE_AND_MEASURE); // CMD_MEASURE
    if(wait_cmd_stat(1) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start measurement");
        return;
    }

// Determine target pixels based on mode
    uint16_t target_pixels = 64; // Default 8x8
    if (mode == CMD_LOAD_CFG_16X16 || mode == CMD_LOAD_CFG_16x16_HIGH_ACCURACY) target_pixels = 256;
    if (mode == CMD_LOAD_CFG_32X32 || mode == CMD_LOAD_CFG_32x32_HIGH_ACCURACY) target_pixels = 1024;

    ESP_LOGI(TAG, "Measurement started. Waiting for frames of %d pixels...", target_pixels);

    while(1) {
        if (read_frame() == ESP_OK) {
            
            // Check if we have received a fully assembled frame!
            if (out_frame.num_pixels == target_pixels) {
                //ESP_LOGI(TAG, "Frame complete! Received %d pixels.", out_frame.num_pixels);
                
                // Example Print: Center pixel of the grid
                //uint16_t center_idx = (target_pixels / 2) + (target_pixels % 2 == 0 ? -1 : 0); // Adjust for even/odd pixel counts
                /*ESP_LOGI(TAG, "Center Pixel Distance: %d mm (Confidence: %d)", 
                         out_frame.pixels[center_idx].distance_mm, 
                         out_frame.pixels[center_idx].confidence);*/ 

                send_frame_binary(&out_frame);


            } else {
                // We just received the first sub-frame (Even rows).
                // It will loop back around immediately to grab the odd rows.
                //ESP_LOGD(TAG, "Sub-frame received, assembling...");
            }
        }
        vTaskDelay(pdMS_TO_TICKS(1)); 
    }
}

