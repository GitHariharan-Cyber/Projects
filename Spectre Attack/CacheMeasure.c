#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>     // For __rdtscp and _mm_clflush
#include <emmintrin.h>     // For _mm_clflush

uint64_t measure_access_time(uint8_t *addr) {
    unsigned int junk = 0;
    uint64_t start, end;

    start = __rdtscp(&junk);      // Serialize + read timestamp
    junk = *addr;                 // Access the address
    end = __rdtscp(&junk);        // Serialize again

    return end - start;
}

int main() {
    uint8_t dummy[256]; // Sample array
    uint64_t access_time;
    FILE *f_hit = fopen("cache_hit_times.txt", "w");
    FILE *f_miss = fopen("cache_miss_times.txt", "w");

    // Ensure dummy is accessed to bring it into the cache
    volatile uint8_t tmp = dummy[42];

    // Measure cache hits
    for (int i = 0; i < 100; i++) {
        access_time = measure_access_time(&dummy[42]);
        fprintf(f_hit, "%lu\n", access_time);
    }

    // Flush from cache
    _mm_clflush(&dummy[42]);

    // Measure cache misses
    for (int i = 0; i < 100; i++) {
        _mm_clflush(&dummy[42]);  // Ensure it's flushed each time
        access_time = measure_access_time(&dummy[42]);
        fprintf(f_miss, "%lu\n", access_time);
    }

    fclose(f_hit);
    fclose(f_miss);

    return 0;
}

