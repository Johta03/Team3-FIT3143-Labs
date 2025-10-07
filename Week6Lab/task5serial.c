#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
static long N = 100000000;
int main(int argc, char* argv[])
{
int i;
double sum = 0.0;
double piVal;
struct timespec start, end;
double time_taken;
// Get current clock time.
clock_gettime(CLOCK_MONOTONIC, &start);
for(i = 0; i < N; i++)
{
sum += 4.0 / (1 + pow((2.0 * i + 1.0)/(2.0 * N), 2));
}
piVal = sum / (double)N;
// Get the clock current time again
// Subtract end from start to get the CPU time used.
clock_gettime(CLOCK_MONOTONIC, &end);
time_taken = (end.tv_sec - start.tv_sec) * 1e9;
time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) *
1e-9;
printf("Calculated Pi value (Serial-AlgoI) = %12.9f\n",
piVal);
 printf("Overall time (s): %lf\n", time_taken); // ts
return 0;
}