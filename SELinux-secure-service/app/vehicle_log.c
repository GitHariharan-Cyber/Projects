#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main() {
    printf("Secure App Running\n");

    FILE *f = fopen("/var/lib/vehicle_log/data.txt", "w");
    if (f) {
        fprintf(f, "SELinux test data\n");
        fclose(f);
        printf("Wrote data.txt\n");
    } else {
        perror("Failed to write data file");
    }

    printf("Trying to read /etc/shadow...\n");
    system("cat /etc/shadow");

    sleep(20);   // keep process alive for observation
    return 0;
}
