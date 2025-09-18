// Serial prime search program
// - Finds all prime numbers strictly less than n
// - Writes primes to an output file, one per line
// - Prints a timing summary (wall-clock, monotonic clock)
//
// Usage:
//   ./serial_primes <n> [output_file]
// Example:
//   ./serial_primes 10000000 serial_primes.txt
//
// Notes:
// - Uses 64-bit types (long long) to safely handle large n
// - Trial division up to floor(sqrt(a)) for each candidate
// - Range searched is [2, n)

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Returns true if a is prime, else false.
// Handles a < 2, even numbers, and performs trial division by odd divisors
// up to floor(sqrt(a)).
bool is_prime(long long a){
    if (a < 2) return false;
    if (a == 2) return true;
    if ((a % 2) == 0) return false;
    long long mid = (long long)sqrt((double)a);
    for (long long i = 3; i <= mid; i += 2){
        if (a % i == 0) return false;
    }
    return true;
}

// Scans [from, to) and writes all primes found to outfile.
// Returns the number of primes found, or -1 if the file cannot be opened.
long long check_for_prime(long long from, long long to, const char *outfile){
    long long count=0;
    FILE *fptr = fopen(outfile, "w");
    if (!fptr) {
        fprintf(stderr, "Failed to open output file: %s\n", outfile);
        return -1;
    }
    for (long long i = from; i < to; i += 1){
        if (is_prime(i)){
            fprintf(fptr, "%lld\n", i);
            count += 1;
        }
    }
    fclose(fptr);
    return count;
}

// Entry point: parses arguments, runs the prime search, measures time, and
// prints a one-line summary.
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <n> [output_file]\n", argv[0]);
        return 1;
    }
    char *endptr = NULL;
    long long n = strtoll(argv[1], &endptr, 10);
    if (endptr == argv[1] || *endptr != '\0' || n < 0) {
        fprintf(stderr, "Invalid n: %s\n", argv[1]);
        return 1;
    }
    const char *outfile = (argc >= 3 ? argv[2] : "serial_primes.txt");

    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

    long long count = check_for_prime(2, n, outfile);

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    if (count < 0) return 1;

    printf("n=%lld, primes=%lld, time=%.6f s, output=%s\n", n, count, elapsed, outfile);
    return 0;
}