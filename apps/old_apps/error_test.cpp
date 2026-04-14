// apps/error_test.cpp
#include "ls/types.hpp"
#include "ls/bc.hpp"
#include "ls/recon.hpp"
#include "ls/time_int.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <functional>

using namespace ls;

// ------- helpers -------
static inline State cons_from_prim(double rho, double u, double p, double gamma) {
    State s;
    s.rho  = rho;
    s.rhou = rho * u;
    const double e_internal = p / (gamma - 1.0);
    const double e_kinetic  = 0.5 * rho * u * u;
    s.E = e_internal + e_kinetic;
    return s;
}

struct Grid {
    int nx_phys{};
    double L{};
    double dx{};
    std::vector<double> xc;  // physical cell centers (no ghosts)

    Grid(int nx, double L_) : nx_phys(nx), L(L_), dx(L_/nx), xc(nx) {
        for (int i = 0; i < nx; ++i) xc[i] = (i + 0.5) * dx;
    }
};

// initialize physical cells, then rely on fill_ghosts() inside the integrator
static void init_isentropic_sine(std::vector<State>& U, const Grid& g,
                                 double gamma, double rho0, double p0, double u0,
                                 double eps)
{
    // U has size nx_phys + 2*NG
    const int nx_total = static_cast<int>(U.size());
    const int nx_phys  = nx_total - 2*NG;

    for (int i = 0; i < nx_phys; ++i) {
        const double x = g.xc[i];
        const double dr = eps * std::sin(2.0*M_PI * x / g.L); // smooth, periodic
        const double rho = rho0 * (1.0 + dr);
        const double p   = p0 * std::pow(rho / rho0, gamma);   // isentropic
        const double u   = u0;                                 // uniform advection
        U[NG + i] = cons_from_prim(rho, u, p, gamma);
    }
}

// run one simulation on a given grid
static void run_case(int nx, double L, double gamma, double CFL, double t_end,
                     BCKind bc, Recon recon,
                     double rho0, double p0, double u0, double eps,
                     std::vector<State>& U_out, Grid& grid_out)
{
    grid_out = Grid(nx, L);
    const int nx_total = nx + 2*NG;
    std::vector<State> U(nx_total);

    // init physical cells
    init_isentropic_sine(U, grid_out, gamma, rho0, p0, u0, eps);

    // time loop: use advance_ssprk3 which handles fill_ghosts + CFL + callbacks
    double t = 0.0;
    auto on_step     = [](double, double, const std::vector<State>&) {};
    auto post_source = [](double, double, std::vector<State>&) {}; // ✅ a no-op callable


    advance_ssprk3(U, t, t_end, gamma, grid_out.dx, CFL, bc, recon, on_step,post_source);

    // return the final state (physical cells only) in U_out
    U_out.resize(nx);
    for (int i = 0; i < nx; ++i) U_out[i] = U[NG + i];
}

// linear periodic interpolation of reference onto coarse centers
static State sample_ref_periodic(const std::vector<State>& Uref, const Grid& Gref, double xq)
{
    // wrap xq to [0, L)
    double x = std::fmod(xq, Gref.L);
    if (x < 0.0) x += Gref.L;

    // cell index in reference grid
    const double r = (x / Gref.dx) - 0.5; // since centers at (j+0.5)*dx
    int j = static_cast<int>(std::floor(r));
    double a = r - j; // fractional part in [0,1)

    auto wrap = [&](int k) {
        // wrap to [0, nx_ref-1]
        int n = static_cast<int>(Uref.size());
        int kk = ((k % n) + n) % n;
        return kk;
    };

    int j0 = wrap(j);
    int j1 = wrap(j + 1);

    // linear blend
    State s;
    s.rho  = (1.0 - a) * Uref[j0].rho  + a * Uref[j1].rho;
    s.rhou = (1.0 - a) * Uref[j0].rhou + a * Uref[j1].rhou;
    s.E    = (1.0 - a) * Uref[j0].E    + a * Uref[j1].E;
    return s;
}

struct Errors {
    double l2_rho{0.0}, l2_rhou{0.0}, l2_E{0.0}, l2_total{0.0};
};

static Errors compute_l2_errors_vs_reference(const std::vector<State>& Uc, const Grid& Gc,
                                             const std::vector<State>& Uref, const Grid& Gref)
{
    const int n = static_cast<int>(Uc.size());
    double e_rho = 0.0, e_rhou = 0.0, e_E = 0.0;

    for (int i = 0; i < n; ++i) {
        const double x = Gc.xc[i];
        const State s_ref = sample_ref_periodic(Uref, Gref, x);
        const double dr   = Uc[i].rho  - s_ref.rho;
        const double dm   = Uc[i].rhou - s_ref.rhou;
        const double dE   = Uc[i].E    - s_ref.E;
        e_rho  += dr*dr;
        e_rhou += dm*dm;
        e_E    += dE*dE;
    }

    Errors E;
    E.l2_rho  = std::sqrt(e_rho  / n);
    E.l2_rhou = std::sqrt(e_rhou / n);
    E.l2_E    = std::sqrt(e_E    / n);
    // aggregate (simple Euclidean combo of the three components)
    E.l2_total = std::sqrt(E.l2_rho*E.l2_rho + E.l2_rhou*E.l2_rhou + E.l2_E*E.l2_E);
    return E;
}

int main()
{
    // ----- configuration -----
    const double L     = 1.0;
    const double gamma = 1.4;
    const double CFL   = 0.4;
    const double T_end = 0.2;            // short time to remain smooth and avoid steepening
    const BCKind bc    = BCKind::Periodic;
    const Recon  recon = Recon::WENO3;   // test third-order recon

    // Isentropic, smooth IC parameters
    const double rho0 = 1.0;
    const double p0   = 1.0;
    const double u0   = 0.5;   // advects the wave without shocks
    const double eps  = 1e-2;  // small perturbation

    // grids to test (coarse -> fine). The last one is the reference.
    std::vector<int> N_list = { 50, 100, 200, 400, 800, 1600};
    // ----- run reference -----
    const int Nref = N_list.back();
    std::vector<State> Uref;
    Grid Gref(Nref, L);
    run_case(Nref, L, gamma, CFL, T_end, bc, recon, rho0, p0, u0, eps, Uref, Gref);

    // ----- run other grids & compute errors vs reference -----
    std::ofstream csv("error_convergence.csv");
    csv << "N,dx,L2_rho,L2_rhou,L2_E,L2_total\n";
    std::cout << std::scientific << std::setprecision(6);

    for (size_t k = 0; k + 1 < N_list.size(); ++k) { // exclude the reference row
        int N = N_list[k];
        std::vector<State> Uc;
        Grid Gc(N, L);

        run_case(N, L, gamma, CFL, T_end, bc, recon, rho0, p0, u0, eps, Uc, Gc);

        Errors E = compute_l2_errors_vs_reference(Uc, Gc, Uref, Gref);
        csv << N << "," << Gc.dx << ","
            << E.l2_rho << "," << E.l2_rhou << "," << E.l2_E << "," << E.l2_total << "\n";

        std::cout << "N=" << std::setw(4) << N << "  dx=" << Gc.dx
                  << "  L2_total=" << E.l2_total
                  << "  (rho=" << E.l2_rho << ", rhou=" << E.l2_rhou << ", E=" << E.l2_E << ")\n";
    }
    csv.close();

    // Also write pairwise observed orders between successive grids (coarse->finer)
    // using L2_total. This is just printed; plot in Python from the CSV if you like.
    std::cout << "\nObserved order (using L2_total between successive grids):\n";
    // gather total errors for convenience
    std::vector<double> dxs, errs;
    {
        std::ifstream fin("error_convergence.csv");
        std::string hdr; std::getline(fin, hdr);
        int Ntmp; double dxt, e1, e2, e3, etot;
        while (fin >> Ntmp) {
            char comma;
            fin >> comma >> dxt >> comma >> e1 >> comma >> e2 >> comma >> e3 >> comma >> etot;
            dxs.push_back(dxt);
            errs.push_back(etot);
        }
    }
    for (size_t i = 0; i + 1 < errs.size(); ++i) {
        double p = std::log(errs[i] / errs[i+1]) / std::log(dxs[i] / dxs[i+1]);
        std::cout << "  p(" << std::setw(4) << N_list[i] << "→" << std::setw(4) << N_list[i+1]
                  << ") = " << p << "\n";
    }

    std::cout << "\nWrote error_convergence.csv\n";
    return 0;
}
