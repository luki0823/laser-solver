#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "ls/external/toml.hpp"
#include "ls/types.hpp"
#include "ls/mesh.hpp"
#include "ls/eos.hpp"
#include "ls/flux.hpp"
#include "ls/bc.hpp"
#include "ls/recon.hpp"
#include "ls/fv_update.hpp"
#include "ls/time_int.hpp"
#include "ls/io.hpp"

using namespace ls;

// -----------------------------------------------------------------------------
// 1D Config for shock tube / WENO testing
// -----------------------------------------------------------------------------
struct Config1D {
    // --- Simulation parameters ---
    int    nx{200};
    double L{1.0};
    double CFL{0.3};
    double t_end{0.003};
    double gamma{1.4};
    std::string bc{"ReflectLeftCopy"};   // "ReflectLeftCopy" or "CopyEnds"
    std::string recon{"WENO3"};          // "WENO3" or "FirstOrder"

    // --- Initial condition (Sod) ---
    std::string ic_type{"sod"};
    double rho_L{1.0};
    double p_L{100000.0};
    double rho_R{0.125};
    double p_R{10000.0};
    double x_split{0.5};

    // --- Output ---
    std::string out_folder{"data/out"};
    int         out_interval{50};
};

// -----------------------------------------------------------------------------
// Load 1D config from TOML file
// -----------------------------------------------------------------------------
Config1D load_config_1d(const std::string& path)
{
    Config1D c;
    auto tbl = toml::parse(path);

    // Top-level scalars
    c.nx   = toml::find_or(tbl, "nx",   c.nx);
    c.L    = toml::find_or(tbl, "L",    c.L);
    c.CFL  = toml::find_or(tbl, "CFL",  c.CFL);
    c.t_end= toml::find_or(tbl, "t_end",c.t_end);
    c.gamma= toml::find_or(tbl, "gamma",c.gamma);
    c.bc   = toml::find_or(tbl, "bc",   c.bc);
    c.recon= toml::find_or(tbl, "reconstruction", c.recon);

    // Initial condition block
    if (tbl.contains("initial_condition")) {
        auto ic = toml::find(tbl, "initial_condition");
        c.ic_type = toml::find_or(ic, "type", c.ic_type);
        c.rho_L   = toml::find_or(ic, "rho_L", c.rho_L);
        c.p_L     = toml::find_or(ic, "p_L",   c.p_L);
        c.rho_R   = toml::find_or(ic, "rho_R", c.rho_R);
        c.p_R     = toml::find_or(ic, "p_R",   c.p_R);
        c.x_split = toml::find_or(ic, "x_split", c.x_split);
    }

    // Output block
    if (tbl.contains("output")) {
        auto out = toml::find(tbl, "output");
        c.out_folder   = toml::find_or(out, "folder",   c.out_folder);
        c.out_interval = toml::find_or(out, "interval", c.out_interval);
    }

    return c;
}

// -----------------------------------------------------------------------------
// Map your 1D "bc" string (ReflectLeftCopy / CopyEnds) to Bc1D
// -----------------------------------------------------------------------------
Bc1D make_bc1d_from_string_shocktube(const std::string& bc_name)
{
    std::string s = bc_name;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    // Your original naming:
    //   - "ReflectLeftCopy" → left boundary reflective, right boundary copy/outflow
    //   - "CopyEnds"        → copy on both ends (we map to Outflow/Outflow)
    if (s == "reflectleftcopy") {
        return Bc1D{BcType::Wall, BcType::Outflow};
    }
    if (s == "copyends") {
        return Bc1D{BcType::Outflow, BcType::Outflow};
    }

    // Fallback: treat string as two-side spec, e.g. "Wall", "Outflow", etc.
    // (This lets you use "Wall", "Outflow", "Symmetry" directly if you want.)
    return make_bc1d_from_strings(bc_name, bc_name);
}

// -----------------------------------------------------------------------------
// Initialize Sod shock tube in 1D
// -----------------------------------------------------------------------------
void init_sod_1d(
    std::vector<State1D>& U,
    const Mesh1D& mesh,
    const Config1D& c,
    const EOSIdealGas& eos)
{
    const int nx_tot = mesh.nx_tot;

    for (int I = 0; I < nx_tot; ++I) {
        double x = mesh.xc(I);

        double rho, p, u;
        if (x < c.x_split * c.L) {
            rho = c.rho_L;
            p   = c.p_L;
        } else {
            rho = c.rho_R;
            p   = c.p_R;
        }
        u = 0.0;

        double E = p / (eos.gamma - 1.0) + 0.5 * rho * u * u;
        U[I] = State1D{rho, rho*u, E};
    }
}

// -----------------------------------------------------------------------------
// 1D WENO / Shock tube tester main
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // 1) Load config
    std::string cfg = (argc > 1 ? argv[1] : "configs/shocktube.toml");
    Config1D c = load_config_1d(cfg);

    std::cout << "Loaded 1D config from " << cfg << "\n";
    std::cout << "  nx=" << c.nx
              << " L=" << c.L
              << " CFL=" << c.CFL
              << " t_end=" << c.t_end
              << " gamma=" << c.gamma << "\n";
    std::cout << "  bc=" << c.bc
              << " recon=" << c.recon << "\n";

    // 2) Setup EOS and reconstruction type
    EOSIdealGas eos;
    eos.gamma = c.gamma;

    ReconType recon_type = parse_recon_type(c.recon);
    int NG = num_ghost_cells(c.recon);   // picks 2 ghosts for WENO3

    // 3) Build mesh and state containers
    Mesh1D mesh(c.nx, c.L, NG);
    int nx_tot = mesh.nx_tot;

    std::vector<State1D> U(nx_tot,   zero_state1d());
    std::vector<State1D> rhs(nx_tot, zero_state1d());

    // 4) Boundary conditions for this test
    Bc1D bc = make_bc1d_from_string_shocktube(c.bc);

    // 5) Initial condition
    if (c.ic_type == "sod" || c.ic_type == "Sod") {
        init_sod_1d(U, mesh, c, eos);
    } else {
        throw std::runtime_error("Unsupported 1D IC type: " + c.ic_type);
    }

    // Apply BC once to set ghost cells
    apply_bc_1d(U, mesh, bc);

    // 6) Time loop with RK3
    double t = 0.0;
    int    step = 0;

    ensure_dir(c.out_folder);
    write_csv_1d(c.out_folder, step, t, mesh, U, eos);
    std::cout << "Wrote step " << step << " at t=" << t << "\n";

    auto has_bad = [&](const std::vector<State1D>& UU) {
        for (int I = mesh.interior_start(); I <= mesh.interior_end(); ++I) {
            const auto& s = UU[I];
            if (!std::isfinite(s.rho) || !std::isfinite(s.E) ||
                !std::isfinite(s.rhou))
                return true;
        }
        return false;
    };

    while (t < c.t_end) {

        // 1) Max wave speed for CFL
        double smax = 1e-12;
        for (int I = mesh.interior_start(); I <= mesh.interior_end(); ++I) {
            const auto& Ui = U[I];
            double rho = Ui.rho;
            double u   = Ui.rhou / rho;
            double c_s = eos.sound_speed(Ui);
            double a   = std::fabs(u) + c_s;
            smax = std::max(smax, a);
        }

        std::cout << "Max Wave Speed = " << smax << "\n";

        // 2) CFL timestep
        double dt = std::min(
            c.CFL * mesh.dx / smax,
            c.t_end - t
        );

        // Some diagnostics: min/max rho and max |u|
        double rho_min = 1e30, rho_max = -1e30;
        int I_rho_min = -1, I_rho_max = -1;
        double umax = -1.0;
        int I_umax = -1;

        for (int I = mesh.interior_start(); I <= mesh.interior_end(); ++I) {
            const auto& Ui = U[I];
            double rho = Ui.rho;
            double u   = Ui.rhou / rho;

            if (rho < rho_min) { rho_min = rho; I_rho_min = I; }
            if (rho > rho_max) { rho_max = rho; I_rho_max = I; }

            double speed = std::fabs(u);
            if (speed > umax) { umax = speed; I_umax = I; }
        }

        std::cout << "step " << step
                  << " dt=" << dt
                  << " rho=[" << rho_min << " at " << I_rho_min
                  << ", " << rho_max << " at " << I_rho_max << "]"
                  << " |u|_max=" << umax << " at " << I_umax
                  << "\n";

        // 3) Advance one RK3 step
        advance_rk3_1d(U, rhs, mesh, eos, bc, recon_type, dt);

        // 4) Update time and step
        t += dt;
        ++step;

        // 5) NaN/Inf check AFTER the update
        if (has_bad(U)) {
            std::cerr << "NaN/Inf detected at t=" << t << "\n";
            break;
        }

        // 6) Output
        if (step % c.out_interval == 0 || t >= c.t_end) {
            write_csv_1d(c.out_folder, step, t, mesh, U, eos);
            std::cout << "Wrote step " << step << " at t=" << t << "\n";
        }
    }

    std::cout << "Done. Final time: " << t << "\n";
    return 0;
}
