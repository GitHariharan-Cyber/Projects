#include <emmintrin.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

// TODO: Define this threshold based on your previous experiments
// Use it for determining the secret
#define CACHE_HIT_THRESHOLD 200

typedef struct datablock {
  uint8_t lpad[508];
  uint8_t dat;
  uint8_t rpad[508];
} DataBlock;

DataBlock array[256];


void victim() {
  uint8_t secret = 50;

  // It is helpful to give the victim time for a couple accesses to the data - this way, we
  // are less likely to be affected by cache-displacement strategies, which we really
  // don't want for this simple experiment.
  for (int i = 0; i < 20; i++) {
    uint8_t temp = array[secret].dat;
    array[secret].dat = 0;
    temp = array[secret].dat;
  }
}


void flushSideChannel() {

	for (int i=0; i<256; i++){
		array[i].dat = 0;
		_mm_clflush(&array[i].dat);
	}

}

uint8_t reloadSideChannel() {
	uint64_t time1, time2;
	int access=0;
	volatile uint8_t *addr;
	int secret = -1;
	uint64_t best_cycles = -1;

	for (int i = 0; i < 256; i++) {
		addr = &array[i].dat;
		time1 = __rdtscp(&access);
		access = *addr;
		time2 = __rdtscp(&access);
		uint64_t cycles = time2 - time1;
		printf("cycles: %lu, index: %d\n", cycles, i);
		if (cycles <= CACHE_HIT_THRESHOLD && cycles < best_cycles) {
			best_cycles = cycles;
			secret = i;
		}
	}

	return secret;
}


int main(int argc, const char** argv) {
  flushSideChannel();
  victim();
  uint8_t secret = reloadSideChannel();

  printf("The secret is %d\n", secret);
}
