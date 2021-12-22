#include <vector>
#include <chrono>
#include <random>
#include <iostream>
#include <cstdint>
#include <immintrin.h>
#include <stdio.h>

// This is the linear naive implementation used as baseline for the optimization benchmark
float dist_baseline(const std::vector<float>& p1, const std::vector<float>& p2) {

    float result = 0;
    unsigned int i = p1.size();
    while (i--) {
        float d = (p1[i] - p2[i]);
        result += d * d;
    }

    // retrieve squared distance (avoid sqrt)
    return result;
}

inline float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read (int d, const float *x)
{
    assert (0 <= d && d < 4);
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
    switch (d) {
      case 3:
        buf[2] = x[2];
      case 2:
        buf[1] = x[1];
      case 1:
        buf[0] = x[0];
    }
    return _mm_load_ps (buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}


float dist_optimized(const std::vector<float>& p1, const std::vector<float>& p2) {
    unsigned int d = p1.size();
    __m256 msum1 = _mm256_setzero_ps();

    const float* x = &(p1[0]);
    const float* y = &(p2[0]);

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        const __m256 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 +=       _mm256_extractf128_ps(msum1, 0);

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        const __m128 a_m_b1 = mx - my;
        msum2 += a_m_b1 * a_m_b1;
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        __m128 a_m_b1 = mx - my;
        msum2 += a_m_b1 * a_m_b1;
    }

    msum2 = _mm_hadd_ps (msum2, msum2);
    msum2 = _mm_hadd_ps (msum2, msum2);
    return  _mm_cvtss_f32 (msum2);
}

int main(void) {

#ifdef __AVX__
    std::cout << "AVX512 present" << std::endl;
    std::cout << "Testing functions" << std::endl;

    std::vector<float> v1 = {1.12, 18};
    std::vector<float> v2 = {2.15, 8};
    std::cout << "norm result = " << dist_optimized(v1, v2) << std::endl; 

#endif

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-10, 10);

    // creating test points
    const unsigned int numPoints = 8 * 1000000;
    std::vector<float> pointsA;
    pointsA.resize(numPoints);
    for(float& point: pointsA) {
        point = static_cast<float>(distribution(generator));
    }

    std::vector<float> pointsB;
    pointsB.resize(numPoints);
    for(float& point: pointsB) {
        point = static_cast<float>(distribution(generator));
    }

    // call the baseline function
    auto start = std::chrono::steady_clock::now();
    auto res_base = dist_baseline(pointsA, pointsB);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<float> diff = end - start;

    std::cout << "[BASELINE], result = " << res_base << " took " << diff.count() << " seconds " << std::endl;

    start = std::chrono::steady_clock::now();
    auto res_opt = dist_optimized(pointsA, pointsB);
    end = std::chrono::steady_clock::now();
    diff = end - start;

    std::cout << "[OPTIMIZED], result = " << res_opt << " took " << diff.count() << " seconds " << std::endl;
    return 0;
}
