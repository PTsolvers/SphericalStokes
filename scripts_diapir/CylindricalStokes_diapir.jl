# DIAPIR - CYLINDRICAL COORDINATES SYSTEM
const USE_GPU = true
const GPU_ID  = 1

saveflag      = true

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using Plots, Printf, Statistics, LinearAlgebra

# SAVING FUNCTIONS ==========================================================================================
function Save_infos(num, lr, lθ, lz, nr, nθ, nz, εnonl, runtime; out="../out_cyl")
    fid=open(out * "/$(num)_infos.inf", "w")
    @printf(fid,"%d %f %f %f %d %d %d %d %d", num, lr, lθ, lz, nr, nθ, nz, εnonl, runtime); close(fid)
end

function Save_phys(num, η0, ρ0, ρ_in, n_exp, τ_C, vr, g, sh, r, radius, dmp; out="../out_cyl")
    fid=open(out * "/$(num)_phys.inf", "w")
    @printf(fid,"%d  %f  %f  %f    %f     %f   %f  %f  %f  %f  %f      %f",
                num, η0, ρ0, ρ_in, n_exp, τ_C, vr, g , sh, r , radius, dmp); close(fid)
end

@static if USE_GPU
    function SaveArray(Aname, A; out="../out_cyl")
        A_tmp = Array(A)
        fname = string(out, "/A_", Aname, ".bin");  fid = open(fname,"w"); write(fid, A_tmp); close(fid)
    end
else
    function SaveArray(Aname, A; out="../out_cyl")
        fname = string(out, "/A_", Aname, ".bin"); fid = open(fname,"w"); write(fid, A); close(fid)
    end
end

# ===========================================================================================================
@parallel function equal!(A::Data.Array, B::Data.Array)
    @all(A) = @all(B)
    return
end

@parallel function multiply!(A::Data.Array, B::Data.Array, fact::Data.Number)
    @all(A) = @all(B)*fact
    return
end

# BOUNDARY CONDITIONS =======================================================================================
@parallel_indices (ix,iy,iz) function bc_r_0!(A::Data.Array)
    if (ix==1         && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = -A[ix+1,iy,iz]    end
    if (ix==size(A,1) && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = -A[ix-1,iy,iz]    end
    return
end

@parallel_indices (ix,iy,iz) function bc_θ_fact!(   A::Data.Array , COORD::Data.Array,
                                                 fact::Data.Number,     r::Data.Number)
    if (ix<=size(A,1) && iy==1         && iz<=size(A,3))    A[ix,iy,iz] = fact*COORD[ix,iy,iz]*r    end
    if (ix<=size(A,1) && iy==size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = fact*COORD[ix,iy,iz]*r    end
    return
end

@parallel_indices (ix,iy,iz) function bc_θ_0!(A::Data.Array)
    if (ix<=size(A,1) && iy==1         && iz<=size(A,3))    A[ix,iy,iz] = -A[ix,iy+1,iz]    end
    if (ix<=size(A,1) && iy==size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = -A[ix,iy-1,iz]    end
    return
end

@parallel_indices (ix,iy,iz) function bc_z_0!(A::Data.Array)
    if (ix<=size(A,1) && iy<=size(A,2) && iz==1        )    A[ix,iy,iz] = 0.0    end
    if (ix<=size(A,1) && iy<=size(A,2) && iz==size(A,3))    A[ix,iy,iz] = 0.0    end
    return
end

@parallel_indices (ix,iy,iz) function bc_z_lin!(   A::Data.Array , COORD::Data.Array,
                                                fact::Data.Number,     r::Data.Number)
    if (ix<=size(A,1) && iy<=size(A,2) && iz==1        )    A[ix,iy,iz] = 2*COORD[ix,iy,iz  ]*r*fact -
                                                                                A[ix,iy,iz+1]    end
    if (ix<=size(A,1) && iy<=size(A,2) && iz==size(A,3))    A[ix,iy,iz] = 2*COORD[ix,iy,iz  ]*r*fact -
                                                                                A[ix,iy,iz-1]    end
    return
end

# COPY BOUNDARIES ===========================================================================================
@parallel_indices (ix,iy,iz) function copy_bc_r!(A::Data.Array)
    if (ix==1         && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = A[ix+1,iy,iz]    end
    if (ix==size(A,1) && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = A[ix-1,iy,iz]    end
    return
end

@parallel_indices (ix,iy,iz) function copy_bc_θ!(A::Data.Array)
    if (ix<=size(A,1) && iy==1         && iz<=size(A,3))    A[ix,iy,iz] = A[ix,iy+1,iz]    end
    if (ix<=size(A,1) && iy==size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = A[ix,iy-1,iz]    end
    return
end

@parallel_indices (ix,iy,iz) function copy_bc_z!(A::Data.Array)
    if (ix<=size(A,1) && iy<=size(A,2) && iz==1        )    A[ix,iy,iz] = A[ix,iy,iz+1]    end
    if (ix<=size(A,1) && iy<=size(A,2) && iz==size(A,3))    A[ix,iy,iz] = A[ix,iy,iz-1]    end
    return
end

# INITIALIZATION ============================================================================================
@parallel_indices (ix,iy,iz) function initialize_inclusion!( A::Data.Array ,      R::Data.Array,
                                                             θ::Data.Array ,      Z::Data.Array,
                                                             r::Data.Number, radius::Data.Number,
                                                            in::Data.Number)
    if (ix<=size(A,1) && iy<=size(A,2) && iz<=size(A,3))
        if (((R[ix,iy,iz]-r)^2 + (θ[ix,iy,iz]*r)^2 + Z[ix,iy,iz]^2) < radius)    A[ix,iy,iz] = in    end
    end
    return
end

@parallel_indices (ix,iy,iz) function initialize_velocity!(   V::Data.Array , COORD::Data.Array,
                                                           fact::Data.Number,     r::Data.Number)
    if (ix<=size(V,1) && iy==1         && iz<=size(V,3))    V[ix,iy,iz] = COORD[ix,iy,iz]*r*fact    end
    if (ix<=size(V,1) && iy==size(V,2) && iz<=size(V,3))    V[ix,iy,iz] = COORD[ix,iy,iz]*r*fact    end
    if (ix<=size(V,1) && iy<=size(V,2) && iz==1        )    V[ix,iy,iz] = COORD[ix,iy,iz]*r*fact    end
    if (ix<=size(V,1) && iy<=size(V,2) && iz==size(V,3))    V[ix,iy,iz] = COORD[ix,iy,iz]*r*fact    end
    return
end

# ITERATION STRATEGY ========================================================================================
@parallel function maxloc!(η_Max::Data.Array, η::Data.Array)
    @inn(η_Max) = @maxloc(η)
    return
end

@parallel function timesteps!(KBDT::Data.Array , GSDT::Data.Array , η_M::Data.Array,
                               dmp::Data.Number, vpdt::Data.Number,  lr::Data.Number)
    @all(KBDT) = dmp * 2.0 * pi * vpdt / lr * @all(η_M)
    @all(GSDT) =       4.0 * pi * vpdt / lr * @all(η_M)
    return
end

@parallel function timesteps2!(DT_R::Data.Array, KBDT::Data.Array, GSDT::Data.Array,
                                  η::Data.Array, vpdt::Data.Number)
    @all(DT_R) = vpdt^2 / (@all(KBDT) + @all(GSDT)/(1.0 + @all(GSDT)/@all(η)))
    return
end

# SOLVER ====================================================================================================
@parallel function compute_P!(D_RR::Data.Array , D_θθ::Data.Array , D_ZZ::Data.Array,
                              D_Rθ::Data.Array , D_RZ::Data.Array , D_θZ::Data.Array,
                              DIVV::Data.Array ,    P::Data.Array ,    R::Data.Array,
                              τ_RR::Data.Array , τ_θθ::Data.Array , τ_ZZ::Data.Array,
                              τ_Rθ::Data.Array , τ_RZ::Data.Array , τ_θZ::Data.Array,
                                VR::Data.Array ,   Vθ::Data.Array ,   VZ::Data.Array,
                              KBDT::Data.Array , GSDT::Data.Array ,    η::Data.Array,
                                dr::Data.Number,   dθ::Data.Number,   dz::Data.Number)
    @all(DIVV) = @d_xa(VR)/dr + 1/@all(R) * @d_ya(Vθ)/dθ + @av_xa(VR)/@all(R) + @d_za(VZ)/dz
    @all(D_RR) =                @d_xa(VR)/dr                      - 1.0/3.0 * @all(DIVV)
    @all(D_θθ) = 1/@all(R)    * @d_ya(Vθ)/dθ + @av_xa(VR)/@all(R) - 1.0/3.0 * @all(DIVV)
    @all(D_ZZ) =                @d_za(VZ)/dz                      - 1.0/3.0 * @all(DIVV)
    @all(D_Rθ) =                @d_xi(Vθ)/dr + 1/@av_xyi(R) * @d_yi(VR)/dθ - @av_xi(Vθ)/@av_xyi(R)
    @all(D_RZ) =                @d_xi(VZ)/dr +                @d_zi(VR)/dz
    @all(D_θZ) = 1/@av_yzi(R) * @d_yi(VZ)/dθ +                @d_zi(Vθ)/dz
    @all(P)    = @all(P)    - @all(KBDT) * @all(DIVV)
    @all(τ_RR) = (@all(τ_RR) +    @all(GSDT) * 2.0 * @all(D_RR))/(1.0 +    @all(GSDT) /    @all(η))
    @all(τ_θθ) = (@all(τ_θθ) +    @all(GSDT) * 2.0 * @all(D_θθ))/(1.0 +    @all(GSDT) /    @all(η))
    @all(τ_ZZ) = (@all(τ_ZZ) +    @all(GSDT) * 2.0 * @all(D_ZZ))/(1.0 +    @all(GSDT) /    @all(η))
    @all(τ_Rθ) = (@all(τ_Rθ) + @av_xyi(GSDT) *       @all(D_Rθ))/(1.0 + @av_xyi(GSDT) / @av_xyi(η))
    @all(τ_RZ) = (@all(τ_RZ) + @av_xzi(GSDT) *       @all(D_RZ))/(1.0 + @av_xzi(GSDT) / @av_xzi(η))
    @all(τ_θZ) = (@all(τ_θZ) + @av_yzi(GSDT) *       @all(D_θZ))/(1.0 + @av_yzi(GSDT) / @av_yzi(η))
    return
end

@parallel function compute_sigma!(σ_RR::Data.Array, σ_θθ::Data.Array, σ_ZZ::Data.Array, P::Data.Array,
                                  τ_RR::Data.Array, τ_θθ::Data.Array, τ_ZZ::Data.Array)
    @all(σ_RR) = -@all(P) + @all(τ_RR)
    @all(σ_θθ) = -@all(P) + @all(τ_θθ)
    @all(σ_ZZ) = -@all(P) + @all(τ_ZZ)
    return
end

@parallel function compute_TII!( τII::Data.Array, τ_RR::Data.Array, τ_θθ::Data.Array, τ_ZZ::Data.Array,
                                τ_Rθ::Data.Array, τ_RZ::Data.Array, τ_θZ::Data.Array)
    @inn(τII) = sqrt(1.0/2.0 * (@inn(τ_RR)^2 + @inn(τ_θθ)^2 + @inn(τ_ZZ)^2) +
                @av_xya(τ_Rθ)^2 + @av_xza(τ_RZ)^2 + @av_yza(τ_θZ)^2)
    return
end

@parallel function power_law!(    η_PL::Data.Array ,     η::Data.Array , τII::Data.Array ,
                              η_PL_OLD::Data.Array , η_INI::Data.Array , τ_C::Data.Number,
                                 n_exp::Data.Number, relax::Data.Number)
    @all(η_PL) = @all(η_INI) * (@all(τII)/τ_C)^(1.0-n_exp)
    @all(η_PL) = exp(log(@all(η_PL))*relax + log(@all(η_PL_OLD))*(1.0-relax))
    @all(η)    = 2.0/(1.0/@all(η_INI) + 1.0/@all(η_PL))
    return
end

@parallel function compute_dV!( dVR::Data.Array ,  dVθ::Data.Array ,  dVZ::Data.Array ,
                               σ_RR::Data.Array , σ_θθ::Data.Array , σ_ZZ::Data.Array ,
                               τ_Rθ::Data.Array , τ_RZ::Data.Array , τ_θZ::Data.Array ,
                                  R::Data.Array ,   ρG::Data.Array ,
                                 dr::Data.Number,   dθ::Data.Number,   dz::Data.Number)
    @all(dVR) = @d_xi(σ_RR)/dr + 1/@av_xi(R) * @d_ya(τ_Rθ)/dθ + @d_za(τ_RZ)/dz +
                    @av_xi(σ_RR)/@av_xi(R) - @av_xi(σ_θθ)/@av_xi(R) - @av_xi(ρG)
    @all(dVθ) = @d_xa(τ_Rθ)/dr + 1/@av_yi(R) * @d_yi(σ_θθ)/dθ + @d_za(τ_θZ)/dz +
                2 * @av_xa(τ_Rθ)/@av_yi(R)
    @all(dVZ) = @d_xa(τ_RZ)/dr + 1/@av_zi(R) * @d_ya(τ_θZ)/dθ + @d_zi(σ_ZZ)/dz +
                    @av_xa(τ_RZ)/@av_zi(R)
    return
end

@parallel function compute_V!( VR::Data.Array,  Vθ::Data.Array,  VZ::Data.Array,
                              dVR::Data.Array, dVθ::Data.Array, dVZ::Data.Array, DT_R::Data.Array)
    @inn(VR) = @inn(VR) + @all(dVR)*@av_xi(DT_R)
    @inn(Vθ) = @inn(Vθ) + @all(dVθ)*@av_yi(DT_R)
    @inn(VZ) = @inn(VZ) + @all(dVZ)*@av_zi(DT_R)
    return
end

# CHECK ERROR ===============================================================================================
@parallel function err_rθ!(err_rθ::Data.Array, τ_Rθ::Data.Array, D_Rθ::Data.Array, η::Data.Array)
    @all(err_rθ) = @all(τ_Rθ) - @all(D_Rθ)*@av_xyi(η)
    return
end

# ===========================================================================================================
@views function CYL_3D_Inclusion()
    num     = 1
    runtime = 0.0
    # physical parameters -----------------------------------------------------------------------------------
    η0          = 1.0                       # Pa*s  , media viscosity
    vr          = 1e2                       #         viscosity ratio out/in
    ρ0          = 0.0                       # kg/m^3, media density
    ρ_in        = ρ0 - 10.0                 # kg/m^3, inclusion density
    g           = 1.0                       # m/s^2 , gravity acceleration
    n_exp       = 1.0                       #         power law exponent
    n_exp_PL    = 5.0                       #         power-law exponent
    τ_C         = 1.0                       # Pa    , xharacteristic stress
    relax       = 1e-3                      #         relaxation parameter (power law)
    r           = 1000.0                    # m     , radius of the total cylinder
    lr          = 6.0                       # m     , model dimension in r
    lθ          = lr/r                      # m     , model dimension in θ
    lz          = 1.0*lr                    # m     , model dimension in z
    radius      = 1.0                       # m     , radius of the inclusion
    sh          = 1.0                       # m/s   , shearing velocity
    # numerics ----------------------------------------------------------------------------------------------
    nr          = 208 - 1                   # number of grid cells in direction r
    nθ          = 208 - 1                   # number of grid cells in direction θ
    nz          = 208 - 1                   # number of grid cells in direction z
    εnonl       = 5e-7                      # pseudo-transient loop exit criteria
    nt          = 1                         # number of time steps
    maxiter     = 1e5                       # maximum number of pseudo-transient iterations
    nout        = 1e2                       # pseudo-transient plotting frequency
    CFL         = 1.0/(2.0 + 4.5*log10(vr)) # Courant-Friedrichs-Lewy condition
    dmp         = 4.5                       # damping parameter
    # preprocessing -----------------------------------------------------------------------------------------
    dr     = lr/(nr-1)                                           # size of cell in direction r
    dθ     = lθ/(nθ-1)                                           # size of cell in direction θ
    dz     = lz/(nz-1)                                           # size of cell in direction z
    R_i    = range(r- lr    /2.0, r+ lr    /2.0, length=nr  )    # coordinates of grid points in r
    θ_i    = range( - lθ    /2.0,    lθ    /2.0, length=nθ  )    # coordinates of grid points in θ
    Z_i    = range( - lz    /2.0,    lz    /2.0, length=nz  )    # coordinates of grid points in z
    R_i_vr = range(r-(lr+dr)/2.0, r+(lr+dr)/2.0, length=nr+1)    # coordinates of grid points in r_vr
    Z_i_vz = range( -(lz+dz)/2.0,   (lz+dz)/2.0, length=nz+1)    # coordinates of grid points in z_vz
    (R   ,θ   ,Z   ) = ([xc for xc = R_i   ,yc = θ_i,zc = Z_i   ],[yc for xc = R_i   ,yc = θ_i,zc = Z_i   ],
                        [zc for xc = R_i   ,yc = θ_i,zc = Z_i   ])    # grid of coordinates
    (R_VR,θ_VR,Z_VR) = ([xc for xc = R_i_vr,yc = θ_i,zc = Z_i   ],[yc for xc = R_i_vr,yc = θ_i,zc = Z_i   ],
                        [zc for xc = R_i_vr,yc = θ_i,zc = Z_i   ])    # grid of coordinates
    (R_VZ,θ_VZ,Z_VZ) = ([xc for xc = R_i   ,yc = θ_i,zc = Z_i_vz],[yc for xc = R_i   ,yc = θ_i,zc = Z_i_vz],
                        [zc for xc = R_i   ,yc = θ_i,zc = Z_i_vz])    # grid of coordinates
    R                = Data.Array(R   )
    θ                = Data.Array(θ   )
    Z                = Data.Array(Z   )
    R_VR             = Data.Array(R_VR)
    θ_VR             = Data.Array(θ_VR)
    Z_VR             = Data.Array(Z_VR)
    R_VZ             = Data.Array(R_VZ)
    θ_VZ             = Data.Array(θ_VZ)
    Z_VZ             = Data.Array(Z_VZ)
    # initialization and boundary conditions ----------------------------------------------------------------
    print("Starting initialization ... ")
    P           =   @zeros(nr  , nθ  , nz  )
    DIVV        =   @zeros(nr  , nθ  , nz  )
    VR          =   @zeros(nr+1, nθ  , nz  )
    Vθ          =   @zeros(nr  , nθ+1, nz  )
    VZ          =   @zeros(nr  , nθ  , nz+1)
    dVR         =   @zeros(nr-1, nθ-2, nz-2)
    dVθ         =   @zeros(nr-2, nθ-1, nz-2)
    dVZ         =   @zeros(nr-2, nθ-2, nz-1)
    VR_OLD      =   @zeros(nr+1, nθ  , nz  )
    Vθ_OLD      =   @zeros(nr  , nθ+1, nz  )
    VZ_OLD      =   @zeros(nr  , nθ  , nz+1)
    D_RR        =   @zeros(nr  , nθ  , nz  )
    D_θθ        =   @zeros(nr  , nθ  , nz  )
    D_ZZ        =   @zeros(nr  , nθ  , nz  )
    D_Rθ        =   @zeros(nr-1, nθ-1, nz-2)
    D_RZ        =   @zeros(nr-1, nθ-2, nz-1)
    D_θZ        =   @zeros(nr-2, nθ-1, nz-1)
    τ_RR        =   @zeros(nr  , nθ  , nz  )
    τ_θθ        =   @zeros(nr  , nθ  , nz  )
    τ_ZZ        =   @zeros(nr  , nθ  , nz  )
    τ_Rθ        =   @zeros(nr-1, nθ-1, nz-2)
    τ_RZ        =   @zeros(nr-1, nθ-2, nz-1)
    τ_θZ        =   @zeros(nr-2, nθ-1, nz-1)
    σ_RR        =   @zeros(nr  , nθ  , nz  )
    σ_θθ        =   @zeros(nr  , nθ  , nz  )
    σ_ZZ        =   @zeros(nr  , nθ  , nz  )
    ρ           = ρ0*@ones(nr  , nθ  , nz  )
    ρG          =   @zeros(nr  , nθ  , nz  )
    η           = η0*@ones(nr  , nθ  , nz  )
    η_INI       =   @zeros(nr  , nθ  , nz  )
    η_PL        =   @zeros(nr  , nθ  , nz  )
    η_PL_OLD    =   @zeros(nr  , nθ  , nz  )
    ηSM         =   @zeros(nr  , nθ  , nz  )
    KBDT        =   @zeros(nr  , nθ  , nz  )
    GSDT        =   @zeros(nr  , nθ  , nz  )
    DT_R        =   @zeros(nr  , nθ  , nz  )
    τII         =   @zeros(nr  , nθ  , nz  )
    err_evo     = []; iter_evo = []
    iters       = 0.0
    ERR_Rθ      = @zeros(nr-1, nθ-1, nz-2)
    @parallel initialize_inclusion!(ρ, R, θ, Z, r, radius, ρ_in)
    @parallel initialize_inclusion!(η, R, θ, Z, r, radius, η0/vr)
    @parallel multiply!(ρG, ρ, g)
    @parallel initialize_velocity!(VZ, θ_VZ, sh, r)
    vpdt        = dr*CFL
    err         = 1.0
    @parallel equal!(η_INI, η)
    @parallel equal!(η_PL , η)
    # action ------------------------------------------------------------------------------------------------
    print("Starting calculation ... \n")
    runtime = @elapsed for it = 1:nt # time loop
        for iter = 1:maxiter # pseudo-transient loop
            # iteration strategy ----------------------------------------------------------------------------
            @parallel equal!(ηSM, η)
            @parallel maxloc!(ηSM, η)
            @parallel copy_bc_r!(ηSM)
            @parallel copy_bc_θ!(ηSM)
            @parallel copy_bc_z!(ηSM)
            @parallel timesteps!(KBDT, GSDT, ηSM, dmp, vpdt, lr)
            @parallel timesteps2!(DT_R, KBDT, GSDT, η, vpdt)
            # SOLVER ----------------------------------------------------------------------------------------
            @parallel compute_P!(D_RR, D_θθ, D_ZZ, D_Rθ, D_RZ, D_θZ, DIVV, P   , R, τ_RR, τ_θθ, τ_ZZ,
                                 τ_Rθ, τ_RZ, τ_θZ, VR  , Vθ  , VZ  , KBDT, GSDT, η, dr  , dθ  , dz  )
            @parallel compute_sigma!(σ_RR, σ_θθ, σ_ZZ, P, τ_RR, τ_θθ, τ_ZZ)
            if err < 1e-5
                n_exp = n_exp_PL
            end
            if n_exp > 1
                @parallel compute_TII!(τII, τ_RR, τ_θθ, τ_ZZ, τ_Rθ, τ_RZ, τ_θZ)
                @parallel copy_bc_r!(τII)
                @parallel copy_bc_θ!(τII)
                @parallel copy_bc_z!(τII)
                @parallel equal!(η_PL_OLD, η_PL)
                @parallel power_law!(η_PL, η, τII, η_PL_OLD, η_INI, τ_C, n_exp, relax)
            end
            @parallel equal!(VR_OLD, VR)
            @parallel equal!(Vθ_OLD, Vθ)
            @parallel equal!(VZ_OLD, VZ)
            @parallel compute_dV!(dVR, dVθ, dVZ, σ_RR, σ_θθ, σ_ZZ, τ_Rθ, τ_RZ, τ_θZ, R, ρG, dr, dθ, dz)
            @parallel compute_V!(VR, Vθ, VZ, dVR, dVθ, dVZ, DT_R)
            @parallel copy_bc_r!(VZ)
            @parallel copy_bc_r!(Vθ)
            @parallel bc_r_0!(VR)
            @parallel bc_θ_fact!(VZ,θ_VZ,sh,r)
            @parallel bc_θ_fact!(VR,θ_VR,0.0,r)
            @parallel bc_θ_0!(Vθ)
            @parallel bc_z_0!(Vθ)
            @parallel bc_z_0!(VR)
            @parallel bc_z_lin!(VZ,θ_VZ,sh,r)
            # pseudo-transient loop exit criteria -----------------------------------------------------------
            if mod(iter,nout) == 0
                err_vr = maximum(abs.(VR[:].-VR_OLD[:]))./maximum(abs.(VR[:]))
                err_vθ = maximum(abs.(Vθ[:].-Vθ_OLD[:]))./maximum(abs.(Vθ[:]))
                err_vz = maximum(abs.(VZ[:].-VZ_OLD[:]))./maximum(abs.(VZ[:]))
                err    = max(err_vr, err_vθ, err_vz)
                @parallel err_rθ!(ERR_Rθ, τ_Rθ, D_Rθ, η)
                err_rθ = maximum(abs.(ERR_Rθ[:]))
                if err<εnonl && iter>20
                    iters = iter
                    break
                end
                # post-processing
                push!(err_evo, err)
                push!(iter_evo,iter)
                @printf("iter %d, err=%1.3e, err_rθ=%1.3e \n", iter, err, err_rθ)
            end
        end
    end
    # SAVING ------------------------------------------------------------------------------------------------
    if saveflag
        !ispath("../out_cyl") && mkdir("../out_cyl")
        err_evo  = Data.Array(err_evo)
        iter_evo = Data.Array(iter_evo)
        Save_phys(num, η0, ρ0, ρ_in, n_exp, τ_C, vr, g, sh, r, radius, dmp)
        Save_infos(num, lr, lθ, lz, nr, nθ, nz, εnonl, runtime)
        SaveArray("R"       , R       )
        SaveArray("TH"      , θ       )
        SaveArray("Z"       , Z       )
        SaveArray("ETAS"    , η       )
        SaveArray("RHO"     , ρ       )
        SaveArray("RHOG"    , ρG      )
        SaveArray("P"       , P       )
        SaveArray("DIVV"    , DIVV    )
        SaveArray("VR"      , VR      )
        SaveArray("VTH"     , Vθ      )
        SaveArray("VZ"      , VZ      )
        SaveArray("D_RR"    , D_RR    )
        SaveArray("D_THTH"  , D_θθ    )
        SaveArray("D_ZZ"    , D_ZZ    )
        SaveArray("D_RTH"   , D_Rθ    )
        SaveArray("D_RZ"    , D_RZ    )
        SaveArray("D_THZ"   , D_θZ    )
        SaveArray("TAU_RR"  , τ_RR    )
        SaveArray("TAU_THTH", τ_θθ    )
        SaveArray("TAU_ZZ"  , τ_ZZ    )
        SaveArray("TAU_RTH" , τ_Rθ    )
        SaveArray("TAU_RZ"  , τ_RZ    )
        SaveArray("TAU_THZ" , τ_θZ    )
        SaveArray("TII"     , τII     )
        # SaveArray("iters"   , iters   )
        SaveArray("err_evo" , err_evo )
        SaveArray("iter_evo", iter_evo)
    end
    return
end

@time CYL_3D_Inclusion()
