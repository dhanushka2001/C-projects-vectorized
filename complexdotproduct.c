#include <stdio.h>              // For printing to console
#include <stdlib.h>             // Standard C library (for random number generator etc.)
#include <immintrin.h>          // AVX, AVX2, FMA, AVX-512, ...
#include <time.h>               // Timers for profiling code performance

#define ALIGN 64               // Align memory to a cacheline boundary (64 bytes for my CPU)
#define vectorLength 8          // 8 float32s per 256-bit vector (8*32=256)

// Credit to Joel Carpenter for the original code and explaining SIMD and vectorization
// using AVX intrinsic functions: https://www.youtube.com/watch?v=AT5nuQQO96o

int main()
{
    // Using the const qualifier in C is a good practice when we want to ensure that
    // some values should remain constant and should not be accidentally modified.
    // https://www.geeksforgeeks.org/const-qualifier-in-c/

    // Number of elements in array
    const int N = 1<<29; // =2^26=67,108,864

    // Why 2^26? Because it's large enough for the timers to be accurate, large
    // enough that it doesn't fit in CPU cache. Also a multiple of 8 which means
    // our 256-vectors will all fit exactly (so can ignore remainder elements).
    // https://www.youtube.com/watch?v=AT5nuQQO96o&t=3288s

    // Allocate N float32s (N*4 bytes) worth of memory aligned to a cacheline boundary (ALIGN)
    float* A = (float*)_aligned_malloc(N * sizeof(float), ALIGN);
    float* B = (float*)_aligned_malloc(N * sizeof(float), ALIGN);  

    // Put some random numbers in the array from -1.0 to 1.0
    printf("Initializing array: %i\n", N);
    srand(0);
    for (int i = 0; i < N; i++)
    {
        float ra = (2.0f * ((float)rand()) / RAND_MAX) - 1.0f;
        float rb = (2.0f * ((float)rand()) / RAND_MAX) - 1.0f;
        A[i] = ra;
        B[i] = rb;
    }
    printf("Initialized\n");

    /*************************** Non-vectorized implementation ****************************/
    printf("Non-vectorized\n");

    // Start the timer
    clock_t begin = clock();

    // Initialize the real and imaginary components of the summation of the a.b* products
    float sumR1 = 0;
    float sumI1 = 0;

    for (int i = 0; i < N; i += 2)
    {
        // Real and imaginary components of A
        float Ar = A[i];
        float Ai = A[i+1];

        // Real and imaginary components of B conjugate
        float Br = B[i];
        float Bi = -B[i+1];

        // Real and imaginary components of the multiplication of A.B*
        float Cr = Ar*Br - Ai*Bi;
        float Ci = Ar*Bi + Ai*Br;

        // Add the product result to the summation
        sumR1 += Cr;
        sumI1 += Ci;
    }
    // Stop the timer
    clock_t end = clock();

    // Calculate runtime
    double dt1 = (double)(end - begin) / CLOCKS_PER_SEC;

    // print result
    printf("sumR=%.6f, sumI=%.6f\n", sumR1, sumI1);
    printf("Elapsed time: %0.15f seconds\n", dt1);


    /*************************** Vectorized implementation ********************************/
    printf("Vectorized\n"); 

    // Start the timer
    clock_t begin2 = clock();

    // Initialize the real and imaginary components of the summation of the a.b* products ("ps"=packed singles)
    __m256 sumr = _mm256_set1_ps(0.0);
    __m256 sumi = _mm256_set1_ps(0.0);

    // A vector with 1 in the real component elements, and -1 in the imaginary.
    // Used for conjugating a complex number
    const __m256 conj = _mm256_set_ps(-1, 1, -1, 1, -1, 1, -1, 1);
    
    // Alias of the float32 arrays A and B, as arrays of 256-bit packed single vectors (__m256)
    // The "(__m256*)" is essential here! Difference between malloc and (int *)malloc in C: https://stackoverflow.com/q/21146981/7875204 
    __m256* a = (__m256*)A;
    __m256* b = (__m256*)B;

    // Number of 256-bit vectors in the array
    const int n = (N / vectorLength);

    // Initialize bConj and bFlip
    __m256 bConj;
    __m256 bFlip;

    // Seperate multiply and add intrinsic functions
    // for (int j = 0; j < n; j++)
    // {
    //     __m256 cr = _mm256_mul_ps(a[j], b[j]); // |ai_3*bi_3|ar_3*br_3|ai_2*bi_2|ar_2*br_2|ai_1*bi_1|ar_1*br_1|ai_0*bi_0|ar_0*br_0|
    //     __m256 bConj = _mm256_mul_ps(b[j], conj); // |-bi_3|br_3|-bi_2|br_2|-bi_1|br_1|-bi_0|br_0|
    //     __m256 bFlip = _mm256_permute_ps(bConj, 0b10110001); // |br_3|-bi_3|br_2|-bi_2|br_1|-bi_1|br_0|-bi_0| <- [2,3,0,1] real and imaginary swapping
    //     __m256 ci = _mm256_mul_ps(a[j], bFlip); // |ai_3*br_3|-ar_3*bi_3|ai_2*br_2|-ar_2*bi_2|ai_1*br_1|-ar_1*bi_1|ai_0*br_0|-ar_0*bi_0|
    //     sumr = _mm256_add_ps(sumr, cr);
    //     sumi = _mm256_add_ps(sumi, ci);
    // }
     
    // Fused multiply-add intrinsic function
    for (int j = 0; j < n; j++)
    {
        sumr = _mm256_fmadd_ps(a[j], b[j], sumr);
        bConj = _mm256_mul_ps(b[j], conj); // |-bi_3|br_3|-bi_2|br_2|-bi_1|br_1|-bi_0|br_0|
        bFlip = _mm256_permute_ps(bConj, 0b10110001); // |br_3|-bi_3|br_2|-bi_2|br_1|-bi_1|br_0|-bi_0| <- [2,3,0,1] real and imaginary swapping
        sumi = _mm256_fmadd_ps(a[j], bFlip, sumi);

        // Manual loop unrolling
        // sumr = _mm256_fmadd_ps(a[j+1], b[j+1], sumr);
        // bConj = _mm256_mul_ps(b[j+1], conj); // |-bi_3|br_3|-bi_2|br_2|-bi_1|br_1|-bi_0|br_0|
        // bFlip = _mm256_permute_ps(bConj, 0b10110001); // |br_3|-bi_3|br_2|-bi_2|br_1|-bi_1|br_0|-bi_0| <- [2,3,0,1] real and imaginary swapping
        // sumi = _mm256_fmadd_ps(a[j+1], bFlip, sumi);
    }

    // alias the vector as a float array
    // The "(float*)" is essential here! Difference between malloc and (int *)malloc in C: https://stackoverflow.com/q/21146981/7875204 
    float* sr = (float*)&sumr;
    float* si = (float*)&sumi;

    // // Lazy mode
    // // real
    // float sumR2 = sr[0] + sr[1] + sr[2] + sr[3] + sr[4] + sr[5] + sr[6] + sr[7];
    // // imaginary
    // float sumI2 = si[0] + si[1] + si[2] + si[3] + si[4] + si[5] + si[6] + si[7];

    // Use horizontal vector addition intrinsics to add up every element in the vector
    // Real
    sumr = _mm256_hadd_ps(sumr, sumr);
    sumr = _mm256_hadd_ps(sumr, sumr);
    sumr = _mm256_add_ps(sumr, _mm256_permute2f128_ps(sumr, sumr, 1));
    // Imaginary
    sumi = _mm256_hadd_ps(sumi, sumi);
    sumi = _mm256_hadd_ps(sumi, sumi);
    sumi = _mm256_add_ps(sumi, _mm256_permute2f128_ps(sumi, sumi, 1));

    float sumR2 = sr[0];
    float sumI2 = si[0];

    // Stop the timer
    clock_t end2 = clock();

    // Free N float32s (N*4 bytes) worth of memory (slow process, do after timer)
    _aligned_free(A);
    _aligned_free(B);

    // Calculate runtime
    double dt2 = (double)(end2 - begin2) / CLOCKS_PER_SEC;

    // print result
    printf("sumR=%.6f, sumI=%.6f\n", sumR2, sumI2);
    printf("Elapsed time: %0.15f seconds\n", dt2);
}

// Test results (with "-ftree-vectorize -funroll-loops -O3" optimization flags):
//
// For N = 2^26:
//  Initializing array: 67108864
//  Non-vectorized
//  sumR=1139.104248, sumI=-3531.591064    
//  Elapsed time: 0.071000000000000 seconds
//  Vectorized
//  sumR=1138.747681, sumI=-3531.189453    
//  Elapsed time: 0.041000000000000 seconds
//
// For N = 2^27 (with optimization flags DISABLED):
//  Initializing array: 134217728
//  Non-vectorized
//  sumR=413.578583, sumI=-8374.754883     
//  Elapsed time: 0.113000000000000 seconds
//  Vectorized
//  sumR=413.302795, sumI=-8374.585938     
//  Elapsed time: 0.086000000000000 seconds
//
// For N = 2^27 (with optimization flags ENABLED):
//  Initializing array: 134217728
//  Non-vectorized
//  sumR=413.578583, sumI=-8374.754883     
//  Elapsed time: 0.130000000000000 seconds
//  Vectorized
//  sumR=413.302795, sumI=-8374.585938     
//  Elapsed time: 0.071000000000000 seconds
//
// For N = 2^30 (caused RAM to go to >90%):
//  Initializing array: 1073741824
//  Non-vectorized
//  sumR=1660.681396, sumI=833.834656
//  Elapsed time: 3.759000000000000 seconds
//  Vectorized
//  sumR=1659.198364, sumI=833.785645
//  Elapsed time: 1.022000000000000 seconds
//
//
// NOTE: would be better to have a test where the exact value is known so we can
//       calculate the % error for vectorization vs non-vectorization.
//       (e.g. approximating Pi)
//
// For some reason, when I comment the non-vectorized code, the vectorized code
// becoems slower...?