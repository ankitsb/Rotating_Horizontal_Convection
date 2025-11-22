using Pkg
using Oceananigans
using Oceananigans.Units
using NCDatasets
using CUDA
using SeawaterPolynomials.TEOS10
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Fields: interpolate
using LinearAlgebra
using Oceananigans.Grids: xnodes, ynodes, znodes#, ConstantToStretchedCoordinate
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, ∂xᶜᶜᶜ, ∂yᶜᶜᶜ, ∂zᶜᶜᶜ, Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.Models: seawater_density
using Oceanostics

const Lz = H = 0.1            # vertical domain extent
const Lx = Ly = 1           # horizontal domain extent
Nx, Ny, Nz = 512, 512, 128    # horizontal, vertical resolution

# chebychev_spaced_y_faces(j) = Ly * (1 - cos(π * (j - 1) / Ny)) / 2
# chebychev_spaced_z_faces(k) = - Lz * (1 + cos(π * (k - 1) / Nz)) / 2

grid = RectilinearGrid(GPU(), size = (Nx, Ny, Nz),
                       x = (0, Lx), y = (0, Ly), #chebychev_spaced_y_faces,
                       z = (-Lz, 0), #chebychev_spaced_z_faces,
                       topology = (Bounded, Bounded, Bounded)
)

@info "Build a grid:"
@show grid

@inline bˢ(x, y, t, p) = p.T_dev * sin(π * (y + p.Ly/2)) + p.T_mean + p.purtur

const purtur = -1.0
const T_mean = 20.0
const T_dev = 10.0
b_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(bˢ, parameters=(; Ly, T_mean, T_dev, purtur)),
                                bottom = GradientBoundaryCondition(0.0),
                                north = GradientBoundaryCondition(0.0),
                                south = GradientBoundaryCondition(0.0),
                                east = GradientBoundaryCondition(0.0),
                                west = GradientBoundaryCondition(0.0)
)

u_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),
                                bottom = ValueBoundaryCondition(0.0),
                                north = ValueBoundaryCondition(0.0),
                                south = ValueBoundaryCondition(0.0),
                                # east = ValueBoundaryCondition(0.0),
                                # west = ValueBoundaryCondition(0.0)
)

v_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),
                                bottom = ValueBoundaryCondition(0.0),
                                # north = ValueBoundaryCondition(0.0),
                                # south = ValueBoundaryCondition(0.0),
                                east = ValueBoundaryCondition(0.0),
                                west = ValueBoundaryCondition(0.0)
)

w_bcs = FieldBoundaryConditions(north = ValueBoundaryCondition(0.0),
                                south = ValueBoundaryCondition(0.0),
                                east = ValueBoundaryCondition(0.0),
                                west = ValueBoundaryCondition(0.0)
)

const Pr = 5      # Prandtl number
const Ra = 1e12   # Rayleigh number

const ν = 1e-6
const κ = ν / Pr

ΔT = T_mean
g_alf = Ra * ν * κ / (ΔT * Lx^2)

const α = 2e-4 # Cat, 2019
const β = 0.0

eos = LinearEquationOfState(thermal_expansion  = α,
                            haline_contraction = β)

const g = g_alf / α
buoyancy = SeawaterBuoyancy(equation_of_state=eos, gravitational_acceleration=g, constant_salinity=34.0)

U = (g * α * ΔT * H)^(1/2)
Ro = 0.04
const f0 = U / (Ro * 1)
beta = f0

coriolis = BetaPlane(f₀=0, β=beta)

vitd = VerticallyImplicitTimeDiscretization()
molecular_diffusivity = ScalarDiffusivity(vitd, ν=ν, κ=(T = κ))

model = NonhydrostaticModel(; grid,
                            advection = WENO(order=9),
                            timestepper = :RungeKutta3,
                            coriolis = coriolis,
                            tracers = (:T, :S),
                            buoyancy = buoyancy,
                            closure = molecular_diffusivity,
                            boundary_conditions = (; T=b_bcs, u=u_bcs, v=v_bcs, w=w_bcs)
)

@info "Constructed a model"
@show model

@inline Ξ(z) = @inbounds randn() * cos(z * π / H) * (1 + z / H) # noise

@inline Tᵢ(x, y, z) = @inbounds T_mean + 1e-3 * Ξ(z)

# Velocity initial condition: random noise scaled by the friction velocity.
@inline vᵢ(x, y, z) = @inbounds 1e-3 * Ξ(z)

# `set!` the `model` fields using functions or constants:
set!(model, v=vᵢ, w=vᵢ, T=Tᵢ, S=34.0)

@info "model initial conditions are Set!!"

simulation = Simulation(model, Δt=1e-3, stop_time=8hour)

conjure_time_step_wizard!(simulation, IterationInterval(1), max_Δt=0.2, diffusive_cfl=0.5, cfl=2.0)#,

using Printf
progress(sim) = @printf("Iter: % 6d, sim time: % 1.3f, wall time: % 10s, Δt: % 1.4f, advective CFL: %.2e, diffusive CFL: %.2e\n",
                        iteration(sim), time(sim), prettytime(sim.run_wall_time),
                        sim.Δt, AdvectiveCFL(sim.Δt)(sim.model), DiffusiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(30))

u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S

direc = "/data/gpfs/projects/punim1661/hori_conv/purturbation_exp/T_m_1/"
filename = "Hori_Conv_v01"
outputs = (; u, v, w, T)

simulation.output_writers[:fields] = NetCDFWriter(model, outputs;
                     filename = joinpath(direc, "output_" * filename * ".nc"),
                     schedule = TimeInterval(5minutes),
                     overwrite_existing = true #false #
)

# u_top(model) = CUDA.@allowscalar model.velocities.u[1:Nx+1, 1:Ny, Nz-5]
# v_top(model) = CUDA.@allowscalar model.velocities.v[1:Nx, 1:Ny+1, Nz-5]
# w_top(model) = CUDA.@allowscalar model.velocities.w[1:Nx, 1:Ny, Nz-5]
# T_top(model) = CUDA.@allowscalar model.tracers.T[1:Nx, 1:Ny, Nz-5]

function KE(model)
    U = @at (Center, Center, Center) model.velocities.u
    V = @at (Center, Center, Center) model.velocities.v
    W = @at (Center, Center, Center) model.velocities.w
    
    KE = Field(0.5 * (U * U + V * V + W * W), indices=(:, :, Nz-5))
    
    return interior(KE)[1:Nx, 1:Ny, 1]
end

# u_x0(model) = CUDA.@allowscalar model.velocities.u[5, 1:Ny, 1:Nz]
# v_x0(model) = CUDA.@allowscalar model.velocities.v[5, 1:Ny+1, 1:Nz]
# w_x0(model) = CUDA.@allowscalar model.velocities.w[5, 1:Ny, 1:Nz+1]
T_x0(model) = CUDA.@allowscalar model.tracers.T[5, 1:Ny, 1:Nz]

# u_x1(model) = CUDA.@allowscalar model.velocities.u[Nx-5, 1:Ny, 1:Nz]
# v_x1(model) = CUDA.@allowscalar model.velocities.v[Nx-5, 1:Ny+1, 1:Nz]
# w_x1(model) = CUDA.@allowscalar model.velocities.w[Nx-5, 1:Ny, 1:Nz+1]
# T_x1(model) = CUDA.@allowscalar model.tracers.T[Nx-5, 1:Ny, 1:Nz]

function psi(model)
    V = Field(Average(model.velocities.v, dims=(1)))
    V_cumsum = Field(Accumulation(cumsum!, V, dims=3))
    
    psi = Field(Average(V, dims=(3))) - V_cumsum
    return interior(psi)[1, 1:Ny+1, 1:Nz]
end

# u_y0(model) = CUDA.@allowscalar model.velocities.u[1:Nx+1, 5, 1:Nz]
# v_y0(model) = CUDA.@allowscalar model.velocities.v[1:Nx, 5, 1:Nz]
# w_y0(model) = CUDA.@allowscalar model.velocities.w[1:Nx, 5, 1:Nz+1]
# T_y0(model) = CUDA.@allowscalar model.tracers.T[1:Nx, 5, 1:Nz]

# u_y1(model) = CUDA.@allowscalar model.velocities.u[1:Nx+1, Ny-5, 1:Nz]
# v_y1(model) = CUDA.@allowscalar model.velocities.v[1:Nx, Ny-5, 1:Nz]
w_y1(model) = CUDA.@allowscalar model.velocities.w[1:Nx, Ny-5, 1:Nz+1]
# T_y1(model) = CUDA.@allowscalar model.tracers.T[1:Nx, Ny-5, 1:Nz]

outputs = Dict(#"u_top" => u_top, "v_top" => v_top, "w_top" => w_top, "T_top" => T_top, 
               # "u_x0" => u_x0, "v_x0" => v_x0, "w_x0" => w_x0, 
                "T_x0" => T_x0, "KE" => KE,
               # "u_x1" => u_x1, "v_x1" => v_x1, "w_x1" => w_x1, "T_x1" => T_x1, 
                "psi" => psi,
               # "u_y0" => u_y0, "v_y0" => v_y0, "w_y0" => w_y0, "T_y0" => T_y0, 
               # "u_y1" => u_y1, "v_y1" => v_y1, 
                "w_y1" => w_y1, #"T_y1" => T_y1, 
    ) #
dims =Dict(#"u_top" => ("x_faa","y_aca",), "v_top" => ("x_caa","y_afa",), "w_top" => ("x_caa","y_aca",), "T_top" => ("x_caa","y_aca",), 
           # "u_x0" => ("y_aca","z_aac",), "v_x0" => ("y_afa","z_aac",), "w_x0" => ("y_aca","z_aaf",), 
            "T_x0" => ("y_aca","z_aac",), "KE" => ("x_caa","y_aca",),
           # "u_x1" => ("y_aca","z_aac",), "v_x1" => ("y_afa","z_aac",), "w_x1" => ("y_aca","z_aaf",), "T_x1" => ("y_aca","z_aac",), 
            "psi" => ("y_afa","z_aac"), 
           # "u_y0" => ("x_faa","z_aac",), "v_y0" => ("x_caa","z_aac",), "w_y0" => ("x_caa","z_aaf",), "T_y0" => ("x_caa","z_aac",), 
           # "u_y1" => ("x_faa","z_aac",), "v_y1" => ("x_caa","z_aac",), 
            "w_y1" => ("x_caa","z_aaf",), #"T_y1" => ("x_caa","z_aac",), 
    ) #
output_attributes = Dict(
    # "u_top"  => Dict("longname" => "u-velocity near top plane", "units" => "m/s"),
    # "v_top" => Dict("longname" => "v-velocity near top plane", "units" => "m/s"),
    # "w_top"   => Dict("longname" => "w-velocity near top plane", "units" => "m/s"),
    "KE"   => Dict("longname" => "Kinetic energy at the top plane", "units" => "m^2/s^2"),
    # "T_top"   => Dict("longname" => "Temperature near top plane", "units" => "°C"),
    # "u_x0"  => Dict("longname" => "u-velocity near west plane", "units" => "m/s"),
    # "v_x0" => Dict("longname" => "v-velocity near west plane", "units" => "m/s"),
    # "w_x0"   => Dict("longname" => "w-velocity near west plane", "units" => "m/s"),
    "T_x0"   => Dict("longname" => "Temperature near west plane", "units" => "°C"),
    # "u_x1"  => Dict("longname" => "u-velocity near east plane", "units" => "m/s"),
    # "v_x1" => Dict("longname" => "v-velocity near east plane", "units" => "m/s"),
    # "w_x1"   => Dict("longname" => "w-velocity near east plane", "units" => "m/s"),
    # "T_x1"   => Dict("longname" => "Temperature near east plane", "units" => "°C"),
    "psi" => Dict("longname" => "Streamfunction zonally averaged", "units" => "m^2/s^2"),
    # "u_y0"  => Dict("longname" => "u-velocity near south plane", "units" => "m/s"),
    # "v_y0" => Dict("longname" => "v-velocity near south plane", "units" => "m/s"),
    # "w_y0"   => Dict("longname" => "w-velocity near south plane", "units" => "m/s"),
    # "T_y0"   => Dict("longname" => "Temperature near south plane", "units" => "°C"),
    # "u_y1"  => Dict("longname" => "u-velocity near north plane", "units" => "m/s"),
    # "v_y1" => Dict("longname" => "v-velocity near north plane", "units" => "m/s"),
    "w_y1"   => Dict("longname" => "w-velocity near north plane", "units" => "m/s"),
    # "T_y1"   => Dict("longname" => "Temperature near north plane", "units" => "°C"),
)

simulation.output_writers[:slices] =
    NetCDFWriter(model, outputs,
                 schedule=TimeInterval(10), 
                 filename=joinpath(direc, "slices_v01.nc"), 
                 dimensions=dims,
                 output_attributes=output_attributes,
                 overwrite_existing = true #false #
)


############ Budget output ###############
# density_field = Average(seawater_density(model), dims=(1, 2))
const density_ref = 1020
function phi_i(model)
    rho_avg = Field(Average((- g * α * model.tracers.T * density_ref / g + density_ref), dims=(1, 2)))
    # phi_i = - g * κ * Lx * Ly * (maximum(rho_avg) - minimum(rho_avg))
    phi_i = CUDA.@allowscalar - g * κ * Lx * Ly * (interior(rho_avg)[1] - interior(rho_avg)[end])
    return phi_i
end

function phi_z(model)
    b = g * α * model.tracers.T
    density_field = - b * density_ref / g + density_ref
    phi_z = CUDA.@allowscalar Field(Integral(g * density_field * model.velocities.w))[1]
    return phi_z
end

ε(model) = CUDA.@allowscalar Field(Integral(KineticEnergyEquation.DissipationRate(model)))[1]
# ε(model) = CUDA.@allowscalar Field(Integral(KineticEnergyEquation.KineticEnergyIsotropicDissipationRate(model)))[1]
# ε(model) = CUDA.@allowscalar Field(Integral(TurbulentKineticEnergyEquation.IsotropicDissipationRate(model)))[1]

outputs = Dict("dissi" => ε, "phi_i" => phi_i, "phi_z" => phi_z) #
dims = Dict("dissi" => (), "phi_i" => (), "phi_z" => ()) #
output_attributes = Dict(
    "dissi"  => Dict("longname" => "Domain-integrated total viscous dissipation", "units" => "m2s-3"),
    "phi_i" => Dict("longname" => "Domain-integrated conversion between the kinetic and potential energy", "units" => "m2s-3"),
    "phi_z"   => Dict("longname" => "Domain-integrated rate at which the centre of mass of the fluid would be raised or lowered by
molecular diffusion", "units" => "m2s-3")
)

simulation.output_writers[:budgets] =
    NetCDFWriter(model, outputs,
                 schedule=IterationInterval(200), 
                 filename=joinpath(direc, "budgets_v01.nc"), 
                 dimensions=dims,
                 output_attributes=output_attributes,
                 overwrite_existing = true #false #
)
############ Budget output ###############

simulation.output_writers[:checkpointer] = Checkpointer(model,
                    schedule = TimeInterval(5minutes),
                    prefix = "model_checkpoint",
                    cleanup=true)

@info "Output files gererated!!!!!!!!!!"

run!(simulation, pickup=true)

# include("code.jl")