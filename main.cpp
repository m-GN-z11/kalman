#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>

#include "FastNoiseLite.h"

template<typename T>
T clamp(T val, T lo, T hi) {
    return (val < lo) ? lo : (val > hi) ? hi : val;
}

// ============================================================
// 1. 向量与矩阵基础
// ============================================================
using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

Matrix zeros(size_t rows, size_t cols) {
    return Matrix(rows, Vector(cols, 0.0));
}

Matrix identity(size_t n) {
    Matrix I = zeros(n, n);
    for (size_t i = 0; i < n; ++i) I[i][i] = 1.0;
    return I;
}

Matrix matMul(const Matrix& A, const Matrix& B) {
    if (A.empty() || B.empty() || A[0].size() != B.size())
        throw std::invalid_argument("Matrix dimension mismatch in multiplication");
    size_t m = A.size(), n = B[0].size(), inner = B.size();
    Matrix C = zeros(m, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t k = 0; k < inner; ++k)
            if (std::abs(A[i][k]) > 1e-15)
                for (size_t j = 0; j < n; ++j)
                    C[i][j] += A[i][k] * B[k][j];
    return C;
}

Vector matVecMul(const Matrix& A, const Vector& x) {
    if (A.empty() || A[0].size() != x.size())
        throw std::invalid_argument("Matrix-vector dimension mismatch");
    size_t m = A.size(), n = x.size();
    Vector y(m, 0.0);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            y[i] += A[i][j] * x[j];
    return y;
}

Matrix transpose(const Matrix& A) {
    if (A.empty()) return Matrix();
    size_t m = A.size(), n = A[0].size();
    Matrix T = zeros(n, m);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            T[j][i] = A[i][j];
    return T;
}

Matrix matAdd(const Matrix& A, const Matrix& B) {
    if (A.size() != B.size() || A[0].size() != B[0].size())
        throw std::invalid_argument("Matrix dimension mismatch in addition");
    size_t m = A.size(), n = A[0].size();
    Matrix C = zeros(m, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Matrix matSub(const Matrix& A, const Matrix& B) {
    if (A.size() != B.size() || A[0].size() != B[0].size())
        throw std::invalid_argument("Matrix dimension mismatch in subtraction");
    size_t m = A.size(), n = A[0].size();
    Matrix C = zeros(m, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

Matrix invert(const Matrix& A) {
    size_t n = A.size();
    if (n == 0 || A[0].size() != n)
        throw std::invalid_argument("Only square matrix can be inverted");
    Matrix aug = zeros(n, 2 * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) aug[i][j] = A[i][j];
        aug[i][n + i] = 1.0;
    }
    for (size_t i = 0; i < n; ++i) {
        size_t pivot = i;
        double maxVal = std::abs(aug[i][i]);
        for (size_t k = i + 1; k < n; ++k)
            if (std::abs(aug[k][i]) > maxVal) { maxVal = std::abs(aug[k][i]); pivot = k; }
        if (maxVal < 1e-14)
            throw std::runtime_error("Matrix is singular or near-singular");
        if (pivot != i) std::swap(aug[i], aug[pivot]);
        double diag = aug[i][i];
        for (size_t j = 0; j < 2 * n; ++j) aug[i][j] /= diag;
        for (size_t k = 0; k < n; ++k) {
            if (k == i) continue;
            double factor = aug[k][i];
            for (size_t j = 0; j < 2 * n; ++j) aug[k][j] -= factor * aug[i][j];
        }
    }
    Matrix inv = zeros(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            inv[i][j] = aug[i][n + j];
    return inv;
}

// ============================================================
// 2. 多层真实传感器噪声生成器
// ============================================================
class SensorNoise {
public:
    struct Config {
        int seed = 0;
        float freq_low = 0.08f;
        float freq_high = 0.8f;
        int octaves = 4;
        float lacunarity = 2.0f;
        float gain = 0.5f;
        float domain_warp_amp = 1.2f;
        double white_std = 0.15;
        double spike_prob = 0.02;
        double spike_magnitude = 4.0;
        double random_walk_std = 0.05;
    };

    explicit SensorNoise(const Config& cfg) : cfg_(cfg) {
        noise_low_.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        noise_low_.SetFractalType(FastNoiseLite::FractalType_FBm);
        noise_low_.SetFractalOctaves(cfg.octaves);
        noise_low_.SetFractalLacunarity(cfg.lacunarity);
        noise_low_.SetFractalGain(cfg.gain);
        noise_low_.SetFrequency(cfg.freq_low);
        noise_low_.SetSeed(cfg.seed);

        noise_high_.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        noise_high_.SetFractalType(FastNoiseLite::FractalType_FBm);
        noise_high_.SetFractalOctaves(2);
        noise_high_.SetFractalLacunarity(2.5f);
        noise_high_.SetFractalGain(0.4f);
        noise_high_.SetFrequency(cfg.freq_high);
        noise_high_.SetSeed(cfg.seed + 1000);

        warp_noise_.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
        warp_noise_.SetFrequency(0.15f);
        warp_noise_.SetSeed(cfg.seed + 2000);

        rng_.seed(cfg.seed + 3000);
    }

    double sample(double t) {
        float tf = static_cast<float>(t);

        float warp_x = warp_noise_.GetNoise(tf, 0.0f) * cfg_.domain_warp_amp;
        float warp_y = warp_noise_.GetNoise(0.0f, tf) * cfg_.domain_warp_amp;

        double drift = noise_low_.GetNoise(tf + warp_x, warp_y);
        double jitter = noise_high_.GetNoise(tf * 1.7f + warp_y, warp_x)
            * (0.5 + 0.5 * std::abs(drift));
        double white = white_dist_(rng_) * cfg_.white_std;

        walk_bias_ += walk_dist_(rng_) * cfg_.random_walk_std;
        walk_bias_ = clamp(walk_bias_, -3.0, 3.0);

        double spike = 0.0;
        if (spike_dist_(rng_) < cfg_.spike_prob) {
            spike = (spike_sign_(rng_) ? 1.0 : -1.0)
                * cfg_.spike_magnitude * cfg_.white_std;
        }

        double raw = drift * 0.60 + jitter * 0.30 + walk_bias_ * 0.15 + white + spike;
        if (std::abs(raw) > 5.0) {
            raw = (raw > 0) ? 5.0 + std::log1p(raw - 5.0)
                : -5.0 - std::log1p(-raw - 5.0);
        }
        return raw;
    }

    void reset() { walk_bias_ = 0.0; }

private:
    Config cfg_;
    FastNoiseLite noise_low_;
    FastNoiseLite noise_high_;
    FastNoiseLite warp_noise_;

    std::mt19937 rng_;
    std::normal_distribution<double> white_dist_{ 0.0, 1.0 };
    std::normal_distribution<double> walk_dist_{ 0.0, 1.0 };
    std::uniform_real_distribution<double> spike_dist_{ 0.0, 1.0 };
    std::bernoulli_distribution spike_sign_{ 0.5 };

    double walk_bias_ = 0.0;
};

// ============================================================
// 3. 卡尔曼滤波器
// ============================================================
struct KalmanFilter {
    static constexpr size_t STATE_DIM = 8;
    static constexpr size_t MEAS_DIM = 4;
    Vector x;
    Matrix P, F, H, Q, R;
};

KalmanFilter kf_init(const Vector& x0, const Matrix& P0,
    const Matrix& Q, const Matrix& R) {
    KalmanFilter kf;
    kf.x = x0;
    kf.P = P0;
    kf.Q = Q;
    kf.R = R;
    kf.H = zeros(KalmanFilter::MEAS_DIM, KalmanFilter::STATE_DIM);
    kf.H[0][0] = 1.0;
    kf.H[1][1] = 1.0;
    kf.H[2][4] = 1.0;
    kf.H[3][6] = 1.0;
    kf.F = identity(KalmanFilter::STATE_DIM);
    return kf;
}

void kf_predict(KalmanFilter& kf, double dt) {
    kf.F = identity(KalmanFilter::STATE_DIM);
    kf.F[0][2] = dt;
    kf.F[1][3] = dt;
    kf.F[4][5] = dt;
    kf.F[6][7] = dt;

    kf.x = matVecMul(kf.F, kf.x);
    Matrix Ft = transpose(kf.F);
    kf.P = matAdd(matMul(matMul(kf.F, kf.P), Ft), kf.Q);
}

void kf_update(KalmanFilter& kf, const Vector& z) {
    if (z.size() != KalmanFilter::MEAS_DIM)
        throw std::invalid_argument("Measurement vector must have size 4");

    Matrix Ht = transpose(kf.H);
    Matrix S = matAdd(matMul(matMul(kf.H, kf.P), Ht), kf.R);
    Matrix Sinv = invert(S);
    Matrix K = matMul(matMul(kf.P, Ht), Sinv);

    Vector Hx = matVecMul(kf.H, kf.x);
    Vector y(KalmanFilter::MEAS_DIM);
    for (size_t i = 0; i < KalmanFilter::MEAS_DIM; ++i) y[i] = z[i] - Hx[i];

    Vector Ky = matVecMul(K, y);
    for (size_t i = 0; i < KalmanFilter::STATE_DIM; ++i) kf.x[i] += Ky[i];

    Matrix I = identity(KalmanFilter::STATE_DIM);
    Matrix KH = matMul(K, kf.H);
    Matrix I_KH = matSub(I, KH);
    kf.P = matMul(I_KH, kf.P);

    for (size_t i = 0; i < KalmanFilter::STATE_DIM; ++i)
        for (size_t j = i + 1; j < KalmanFilter::STATE_DIM; ++j)
            kf.P[i][j] = kf.P[j][i] = 0.5 * (kf.P[i][j] + kf.P[j][i]);
}

// ============================================================
// 4. 主程序
// ============================================================
int main() {
    auto true_cx = [](double t) { return 10.0 + 1.6 * t - 0.66 * t * t + 0.1 * t * t * t - 0.005 * t * t * t * t; };
    auto true_cy = [](double t) { return 20.0 + 2.0 * t - 0.8 * t * t + 0.12 * t * t * t - 0.006 * t * t * t * t; };
    auto true_vx = [](double t) { return 1.6 - 1.32 * t + 0.3 * t * t - 0.02 * t * t * t; };
    auto true_vy = [](double t) { return 2.0 - 1.6 * t + 0.36 * t * t - 0.024 * t * t * t; };
    auto true_area = [](double t) { return 100.0 + 3.0 * t - 1.5 * t * t + 0.2 * t * t * t - 0.01 * t * t * t * t; };
    auto true_darea = [](double t) { return 3.0 - 3.0 * t + 0.6 * t * t - 0.04 * t * t * t; };
    auto true_energy = [](double t) { return 200.0 + 4.0 * t - 2.0 * t * t + 0.3 * t * t * t - 0.015 * t * t * t * t; };
    auto true_denergy = [](double t) { return 4.0 - 4.0 * t + 0.9 * t * t - 0.06 * t * t * t; };

    const double dt = 0.1;
    const size_t steps = 50;         // 前50个点：滤波（测量+更新）
    const size_t forecast_steps = 1; // 最后1个点：纯预测（无测量更新）

    SensorNoise::Config cfg_cx;
    cfg_cx.seed = 1337; cfg_cx.freq_low = 0.05f; cfg_cx.freq_high = 0.6f;
    cfg_cx.white_std = 0.12; cfg_cx.spike_prob = 0.015; cfg_cx.spike_magnitude = 5.0;
    cfg_cx.random_walk_std = 0.03;
    SensorNoise noise_cx(cfg_cx);

    SensorNoise::Config cfg_cy;
    cfg_cy.seed = 1338; cfg_cy.freq_low = 0.06f; cfg_cy.freq_high = 0.7f;
    cfg_cy.white_std = 0.12; cfg_cy.spike_prob = 0.02; cfg_cy.spike_magnitude = 4.0;
    cfg_cy.random_walk_std = 0.04;
    SensorNoise noise_cy(cfg_cy);

    SensorNoise::Config cfg_area;
    cfg_area.seed = 1339; cfg_area.freq_low = 0.1f; cfg_area.freq_high = 1.0f;
    cfg_area.white_std = 0.3; cfg_area.spike_prob = 0.025; cfg_area.spike_magnitude = 3.5;
    cfg_area.random_walk_std = 0.08;
    SensorNoise noise_area(cfg_area);

    SensorNoise::Config cfg_energy;
    cfg_energy.seed = 1340; cfg_energy.freq_low = 0.08f; cfg_energy.freq_high = 0.9f;
    cfg_energy.white_std = 0.25; cfg_energy.spike_prob = 0.03; cfg_energy.spike_magnitude = 3.0;
    cfg_energy.random_walk_std = 0.06;
    SensorNoise noise_energy(cfg_energy);

    Vector x0_true = { true_cx(0), true_cy(0), true_vx(0), true_vy(0),
                      true_area(0), true_darea(0), true_energy(0), true_denergy(0) };
    Vector x0_est = x0_true;
    x0_est[2] += 0.5;
    x0_est[3] -= 0.3;

    Matrix P0 = identity(KalmanFilter::STATE_DIM);
    for (auto& row : P0) for (auto& val : row) val *= 100.0;

    Matrix Q = identity(KalmanFilter::STATE_DIM);
    Q[0][0] = Q[1][1] = 0.1;
    Q[2][2] = Q[3][3] = 0.05;
    Q[4][4] = 0.2;  Q[5][5] = 0.1;
    Q[6][6] = 0.2;  Q[7][7] = 0.1;

    Matrix R = identity(KalmanFilter::MEAS_DIM);
    R[0][0] = 0.5;
    R[1][1] = 0.5;
    R[2][2] = 3.0;
    R[3][3] = 2.5;

    KalmanFilter kf = kf_init(x0_est, P0, Q, R);

    std::ofstream outfile("kalman_output.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error: Unable to create output file kalman_output.txt" << std::endl;
        return 1;
    }
    outfile << "# t true_cx true_cy meas_cx meas_cy filt_cx filt_cy "
        << "true_vx true_vy filt_vx filt_vy "
        << "true_area meas_area filt_area true_energy meas_energy filt_energy is_forecast\n";
    outfile << std::fixed << std::setprecision(6);

    // Phase 1: Filtering with actual measurements (steps 0-50)
    for (size_t i = 0; i <= steps; ++i) {
        double t = i * dt;

        double cx_true = true_cx(t), cy_true = true_cy(t);
        double vx_true = true_vx(t), vy_true = true_vy(t);
        double area_true = true_area(t), energy_true = true_energy(t);

        double n_cx = noise_cx.sample(t) * std::sqrt(R[0][0] / 1.2) * 2.0;
        double n_cy = noise_cy.sample(t) * std::sqrt(R[1][1] / 1.2) * 2.0;
        double n_area = noise_area.sample(t) * std::sqrt(R[2][2] / 1.5) * 2.0;
        double n_energy = noise_energy.sample(t) * std::sqrt(R[3][3] / 1.4) * 2.0;

        double meas_cx = cx_true + n_cx;
        double meas_cy = cy_true + n_cy;
        double meas_area = area_true + n_area;
        double meas_energy = energy_true + n_energy;

        Vector z = { meas_cx, meas_cy, meas_area, meas_energy };

        if (i == 0) kf_update(kf, z);
        else { kf_predict(kf, dt); kf_update(kf, z); }

        outfile << t << " "
            << cx_true << " " << cy_true << " "
            << meas_cx << " " << meas_cy << " "
            << kf.x[0] << " " << kf.x[1] << " "
            << vx_true << " " << vy_true << " "
            << kf.x[2] << " " << kf.x[3] << " "
            << area_true << " " << meas_area << " " << kf.x[4] << " "
            << energy_true << " " << meas_energy << " " << kf.x[6] << " 0\n";
    }

    // Phase 2: Pure prediction, no measurement updates (next 1 point)
    for (size_t j = 1; j <= forecast_steps; ++j) {
        double t = (steps + j) * dt;

        double cx_true = true_cx(t), cy_true = true_cy(t);
        double vx_true = true_vx(t), vy_true = true_vy(t);
        double area_true = true_area(t), energy_true = true_energy(t);

        // Only predict, do not update with measurements
        kf_predict(kf, dt);

        // No measurements available during forecast — output NaN for meas_ fields
        double nan = std::numeric_limits<double>::quiet_NaN();

        outfile << t << " "
            << cx_true << " " << cy_true << " "
            << nan << " " << nan << " "       // no measurements during forecast
            << kf.x[0] << " " << kf.x[1] << " "
            << vx_true << " " << vy_true << " "
            << kf.x[2] << " " << kf.x[3] << " "
            << area_true << " " << nan << " " << kf.x[4] << " "
            << energy_true << " " << nan << " " << kf.x[6] << " 1\n";
    }

    outfile.close();
    std::cout << "Simulation complete. Results written to kalman_output.txt" << std::endl;
    std::cout << "Phase 1: " << (steps + 1) << " points with filtering (measurement + update)" << std::endl;
    std::cout << "Phase 2: " << forecast_steps << " point with pure prediction (no measurement updates)" << std::endl;
    return 0;
}