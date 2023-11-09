# DIAPIR - CARTESIAN COORDINATES SYSTEM
const USE_GPU = true
const GPU_ID  = 0

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
function Save_infos(num, lx, ly, lz, nx, ny, nz, εnonl, runtime; out="../out_cart")
    fid=open(out * "/$(num)_infos.inf", "w")
    @printf(fid,"%d %f %f %f %d %d %d %d %d", num, lx, ly, lz, nx, ny, nz, εnonl, runtime); close(fid)
end

function Save_phys(num, η0, ρ0, ρ_in, n_exp, τ_C, vr, g, sh, radius, dmp; out="../out_cart")
    fid=open(out * "/$(num)_phys.inf", "w")
    @printf(fid,"%d   %f  %f  %f    %f     %f   %f  %f  %f  %f      %f",
                 num, η0, ρ0, ρ_in, n_exp, τ_C, vr, g , sh, radius, dmp); close(fid)
end

@static if USE_GPU
    function SaveArray(Aname, A; out="../out_cart")
        A_tmp = Array(A)
        fname = string(out, "/A_", Aname, ".bin");  fid = open(fname,"w"); write(fid, A_tmp); close(fid)
    end
else
    function SaveArray(Aname, A; out="../out_cart")
        fname = string(out, "/A_", Aname, ".bin"); fid = open(fname,"w"); write(fid, A); close(fid)
    end
end

# ===========================================================================================================
@parallel_indices (ix,iy,iz) function equal!(A::Data.Array, B::Data.Array)
    if (ix<=size(A,1) && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = B[ix,iy,iz]    end
    return
end

@parallel_indices (ix,iy,iz) function multiply!(A::Data.Array, B::Data.Array, fact::Data.Number)
    if (ix<=size(A,1) && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = B[ix,iy,iz]*fact    end
    return
end

# BOUNDARY CONDITIONS =======================================================================================
@parallel_indices (ix,iy,iz) function bc_x_lin!(A::Data.Array, COORD::Data.Array, fact::Data.Number)
    if (ix==1         && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = 2*COORD[ix  ,iy,iz]*fact -
                                                                                A[ix+1,iy,iz]    end
    if (ix==size(A,1) && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = 2*COORD[ix  ,iy,iz]*fact -
                                                                                A[ix-1,iy,iz]    end
    return
end

@parallel_indices (ix,iy,iz) function bc_x_0!(A::Data.Array)
    if (ix==1         && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = 0.0    end
    if (ix==size(A,1) && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = 0.0    end
    return
end

@parallel_indices (ix,iy,iz) function copy_bc_x!(A::Data.Array)
    if (ix==1         && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = A[ix+1,iy,iz]    end
    if (ix==size(A,1) && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = A[ix-1,iy,iz]    end
    return
end

@parallel_indices (ix,iy,iz) function bc_y_fact!(A::Data.Array, COORD::Data.Array, fact::Data.Number)
    if (ix<=size(A,1) && iy==1         && iz<=size(A,3))    A[ix,iy,iz] = fact*COORD[ix,iy,iz]    end
    if (ix<=size(A,1) && iy==size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = fact*COORD[ix,iy,iz]    end
    return
end

@parallel_indices (ix,iy,iz) function bc_y_0!(A::Data.Array)
    if (ix<=size(A,1) && iy==1         && iz<=size(A,3))    A[ix,iy,iz] = -A[ix,iy+1,iz]    end
    if (ix<=size(A,1) && iy==size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = -A[ix,iy-1,iz]    end
    return
end

@parallel_indices (ix,iy,iz) function bc_z_0!(A::Data.Array)
    if (ix<=size(A,1) && iy<=size(A,2) && iz==1        )    A[ix,iy,iz] = -A[ix,iy,iz+1]    end
    if (ix<=size(A,1) && iy<=size(A,2) && iz==size(A,3))    A[ix,iy,iz] = -A[ix,iy,iz-1]    end
    return
end

@parallel_indices (ix,iy,iz) function copy_bc_z!(A::Data.Array)
    if (ix<=size(A,1) && iy<=size(A,2) && iz==1        )    A[ix,iy,iz] = A[ix,iy,iz+1]    end
    if (ix<=size(A,1) && iy<=size(A,2) && iz==size(A,3))    A[ix,iy,iz] = A[ix,iy,iz-1]    end
    return
end

# COPY BOUNDARIES ===========================================================================================
@parallel_indices (ix,iy,iz) function copy_boundary!(A::Data.Array)
    if (ix==1           && iy<=size(A,2)-2 && iz<=size(A,3)-2)  A[ix  ,iy+1,iz+1] = A[ix+1,iy+1,iz+1]   end
    if (ix==size(A,1)   && iy<=size(A,2)-2 && iz<=size(A,3)-2)  A[ix  ,iy+1,iz+1] = A[ix-1,iy+1,iz+1]   end
    if (ix<=size(A,1)-2 && iy==1           && iz<=size(A,3)-2)  A[ix+1,iy  ,iz+1] = A[ix+1,iy+1,iz+1]   end
    if (ix<=size(A,1)-2 && iy==size(A,2)   && iz<=size(A,3)-2)  A[ix+1,iy  ,iz+1] = A[ix+1,iy-1,iz+1]   end
    if (ix<=size(A,1)-2 && iy<=size(A,2)-2 && iz==1          )  A[ix+1,iy+1,iz  ] = A[ix+1,iy+1,iz+1]   end
    if (ix<=size(A,1)-2 && iy<=size(A,2)-2 && iz==size(A,3)  )  A[ix+1,iy+1,iz  ] = A[ix+1,iy+1,iz-1]   end
    return
end

# INITIALIZATION ============================================================================================
@parallel_indices (ix,iy,iz) function initialize_inclusion!(     A::Data.Array , X3D::Data.Array,
                                                               Y3D::Data.Array , Z3D::Data.Array,
                                                            radius::Data.Number,  in::Data.Number)
    if (ix<=size(A,1) && iy<=size(A,2) && iz<=size(A,3))
        if ((X3D[ix,iy,iz]^2 + Y3D[ix,iy,iz]^2 + Z3D[ix,iy,iz]^2) < radius)    A[ix,iy,iz] = in    end
    end
    return
end

@parallel_indices (ix,iy,iz) function initialize_velocity!(   V::Data.Array, COORD::Data.Array,
                                                           fact::Data.Number)
    if (ix<=size(V,1)-1 && iy<=size(V,2) && iz<=size(V,3))    V[ix,iy,iz] = COORD[ix  ,iy,iz]*fact    end
    if (ix==size(V,1)   && iy<=size(V,2) && iz<=size(V,3))    V[ix,iy,iz] = COORD[ix-1,iy,iz]*fact    end
    return
end

# ITERATION STRATEGY ========================================================================================
@parallel function maxloc!(η_Max::Data.Array, η::Data.Array)
    @inn(η_Max) = @maxloc(η)
    return
end

macro KBDT(ix,iy,iz)    esc(:(    dmp * 2.0 * pi * vpdt / lx * ηSM[$ix,$iy,$iz]    ))    end
macro GSDT(ix,iy,iz)    esc(:(          4.0 * pi * vpdt / lx * ηSM[$ix,$iy,$iz]    ))    end
@parallel_indices (ix,iy,iz) function timesteps!(DT_R::Data.Array ,   η::Data.Array , ηSM::Data.Array,
                                                 vpdt::Data.Number, dmp::Data.Number,  lx::Data.Number)
    if (ix<=size(DT_R,1) && iy<=size(DT_R,2) && iz<=size(DT_R,3))
        DT_R[ix,iy,iz] = vpdt^2 / (@KBDT(ix,iy,iz) + @GSDT(ix,iy,iz)/(1.0 + @GSDT(ix,iy,iz)/η[ix,iy,iz]))
    end
    return
end

# SOLVER ====================================================================================================
macro avxyi_η(ix,iy,iz)     esc(:(    0.25*(  η[$ix  ,$iy  ,$iz+1] +   η[$ix  ,$iy+1,$iz+1] +
                                              η[$ix+1,$iy  ,$iz+1] +   η[$ix+1,$iy+1,$iz+1])    ))   end
macro avxzi_η(ix,iy,iz)     esc(:(    0.25*(  η[$ix  ,$iy+1,$iz  ] +   η[$ix  ,$iy+1,$iz+1] +
                                              η[$ix+1,$iy+1,$iz  ] +   η[$ix+1,$iy+1,$iz+1])    ))   end
macro avyzi_η(ix,iy,iz)     esc(:(    0.25*(  η[$ix+1,$iy  ,$iz  ] +   η[$ix+1,$iy  ,$iz+1] +
                                              η[$ix+1,$iy+1,$iz  ] +   η[$ix+1,$iy+1,$iz+1])    ))   end
macro avxyi_ηSM(ix,iy,iz)   esc(:(    0.25*(ηSM[$ix  ,$iy  ,$iz+1] + ηSM[$ix  ,$iy+1,$iz+1] +
                                            ηSM[$ix+1,$iy  ,$iz+1] + ηSM[$ix+1,$iy+1,$iz+1])    ))   end
macro avxzi_ηSM(ix,iy,iz)   esc(:(    0.25*(ηSM[$ix  ,$iy+1,$iz  ] + ηSM[$ix  ,$iy+1,$iz+1] +
                                            ηSM[$ix+1,$iy+1,$iz  ] + ηSM[$ix+1,$iy+1,$iz+1])    ))   end
macro avyzi_ηSM(ix,iy,iz)   esc(:(    0.25*(ηSM[$ix+1,$iy  ,$iz  ] + ηSM[$ix+1,$iy  ,$iz+1] +
                                            ηSM[$ix+1,$iy+1,$iz  ] + ηSM[$ix+1,$iy+1,$iz+1])    ))   end
macro avxyi_GSDT(ix,iy,iz)  esc(:(    4.0 * pi * vpdt / lx * @avxyi_ηSM(ix,iy,iz)    ))              end
macro avxzi_GSDT(ix,iy,iz)  esc(:(    4.0 * pi * vpdt / lx * @avxzi_ηSM(ix,iy,iz)    ))              end
macro avyzi_GSDT(ix,iy,iz)  esc(:(    4.0 * pi * vpdt / lx * @avyzi_ηSM(ix,iy,iz)    ))              end
macro D_XX(ix,iy,iz)        esc(:(    (VX[$ix+1,$iy  ,$iz  ]-VX[$ix,$iy,$iz])*_dx    ))              end
macro D_YY(ix,iy,iz)        esc(:(    (VY[$ix  ,$iy+1,$iz  ]-VY[$ix,$iy,$iz])*_dy    ))              end
macro D_ZZ(ix,iy,iz)        esc(:(    (VZ[$ix  ,$iy  ,$iz+1]-VZ[$ix,$iy,$iz])*_dz    ))              end
macro D_XY(ix,iy,iz)        esc(:(    (VX[$ix+1,$iy+1,$iz+1]-VX[$ix+1,$iy  ,$iz+1])*_dy +
                                      (VY[$ix+1,$iy+1,$iz+1]-VY[$ix  ,$iy+1,$iz+1])*_dx    ))        end
macro D_XZ(ix,iy,iz)        esc(:(    (VX[$ix+1,$iy+1,$iz+1]-VX[$ix+1,$iy+1,$iz  ])*_dz +
                                      (VZ[$ix+1,$iy+1,$iz+1]-VZ[$ix  ,$iy+1,$iz+1])*_dx    ))        end
macro D_YZ(ix,iy,iz)        esc(:(    (VY[$ix+1,$iy+1,$iz+1]-VY[$ix+1,$iy+1,$iz  ])*_dz +
                                      (VZ[$ix+1,$iy+1,$iz+1]-VZ[$ix+1,$iy  ,$iz+1])*_dy    ))        end
@parallel_indices (ix,iy,iz) function compute_P!(   P::Data.Array , τ_XX::Data.Array , τ_YY::Data.Array ,
                                                 τ_ZZ::Data.Array , τ_XY::Data.Array , τ_XZ::Data.Array ,
                                                 τ_YZ::Data.Array ,   VX::Data.Array ,   VY::Data.Array ,
                                                   VZ::Data.Array ,    η::Data.Array ,  ηSM::Data.Array ,
                                                  dmp::Data.Number, vpdt::Data.Number,   lx::Data.Number,
                                                  _dx::Data.Number,  _dy::Data.Number,  _dz::Data.Number)
    if (ix<=size(P   ,1) && iy<=size(P   ,2) && iz<=size(P   ,3))
        P[ix,iy,iz]    = P[ix,iy,iz]    - @KBDT(ix,iy,iz) * (@D_XX(ix,iy,iz) + @D_YY(ix,iy,iz) +
                                                                               @D_ZZ(ix,iy,iz))      end
    if (ix<=size(τ_XX,1) && iy<=size(τ_XX,2) && iz<=size(τ_XX,3))
        τ_XX[ix,iy,iz] = (τ_XX[ix,iy,iz] + @GSDT(ix,iy,iz)*2.0*@D_XX(ix,iy,iz))/
                                    (1.0 + @GSDT(ix,iy,iz)/η[ix,iy,iz])                              end
    if (ix<=size(τ_YY,1) && iy<=size(τ_YY,2) && iz<=size(τ_YY,3))
        τ_YY[ix,iy,iz] = (τ_YY[ix,iy,iz] + @GSDT(ix,iy,iz)*2.0*@D_YY(ix,iy,iz))/
                                    (1.0 + @GSDT(ix,iy,iz)/η[ix,iy,iz])                              end
    if (ix<=size(τ_ZZ,1) && iy<=size(τ_ZZ,2) && iz<=size(τ_ZZ,3))
        τ_ZZ[ix,iy,iz] = (τ_ZZ[ix,iy,iz] + @GSDT(ix,iy,iz)*2.0*@D_ZZ(ix,iy,iz))/
                                    (1.0 + @GSDT(ix,iy,iz)/η[ix,iy,iz])                              end
    if (ix<=size(τ_XY,1) && iy<=size(τ_XY,2) && iz<=size(τ_XY,3))
        τ_XY[ix,iy,iz] = (τ_XY[ix,iy,iz] + @avxyi_GSDT(ix,iy,iz)*@D_XY(ix,iy,iz))/
                                    (1.0 + @avxyi_GSDT(ix,iy,iz)/@avxyi_η(ix,iy,iz))                 end
    if (ix<=size(τ_XZ,1) && iy<=size(τ_XZ,2) && iz<=size(τ_XZ,3))
        τ_XZ[ix,iy,iz] = (τ_XZ[ix,iy,iz] + @avxzi_GSDT(ix,iy,iz)*@D_XZ(ix,iy,iz))/
                                    (1.0 + @avxzi_GSDT(ix,iy,iz)/@avxzi_η(ix,iy,iz))                 end
    if (ix<=size(τ_YZ,1) && iy<=size(τ_YZ,2) && iz<=size(τ_YZ,3))
        τ_YZ[ix,iy,iz] = (τ_YZ[ix,iy,iz] + @avyzi_GSDT(ix,iy,iz)*@D_YZ(ix,iy,iz))/
                                    (1.0 + @avyzi_GSDT(ix,iy,iz)/@avyzi_η(ix,iy,iz))                 end
    return
end

@parallel_indices (ix,iy,iz) function compute_TII!( τII::Data.Array, τ_XX::Data.Array, τ_YY::Data.Array,
                                                   τ_ZZ::Data.Array, τ_XY::Data.Array, τ_XZ::Data.Array,
                                                   τ_YZ::Data.Array)
    if (ix<=size(τII,1)-2 && iy<=size(τII,2)-2 && iz<=size(τII,3)-2)
        τII[ix+1,iy+1,iz+1] = sqrt(1.0/2.0 * (τ_XX[ix+1,iy+1,iz+1]^2  + τ_YY[ix+1,iy+1,iz+1]^2 +
                                              τ_ZZ[ix+1,iy+1,iz+1]^2) +
                                       (0.25*(τ_XY[ix  ,iy  ,iz  ] + τ_XY[ix  ,iy+1,iz  ] +
                                              τ_XY[ix+1,iy  ,iz  ] + τ_XY[ix+1,iy+1,iz  ]))^2 +
                                       (0.25*(τ_XZ[ix  ,iy  ,iz  ] + τ_XZ[ix  ,iy,  iz+1] +
                                              τ_XZ[ix+1,iy  ,iz  ] + τ_XZ[ix+1,iy  ,iz+1]))^2 +
                                       (0.25*(τ_YZ[ix  ,iy  ,iz  ] + τ_YZ[ix  ,iy  ,iz+1] +
                                              τ_YZ[ix  ,iy+1,iz  ] + τ_YZ[ix  ,iy+1,iz+1]))^2)       end
    return
end

@parallel_indices (ix,iy,iz) function power_law!(η_PL::Data.Array ,        η::Data.Array ,
                                                  τII::Data.Array , η_PL_OLD::Data.Array ,
                                                η_INI::Data.Array ,      τ_C::Data.Number,
                                                n_exp::Data.Number,    relax::Data.Number)
    if (ix<=size(η_PL,1) && iy<=size(η_PL,2) && iz<=size(η_PL,3))
        η_PL[ix,iy,iz] = η_INI[ix,iy,iz] * (τII[ix,iy,iz]/τ_C)^(1.0-n_exp)                      end
    if (ix<=size(η_PL,1) && iy<=size(η_PL,2) && iz<=size(η_PL,3))
        η_PL[ix,iy,iz] = exp(log(η_PL[ix,iy,iz])*relax + log(η_PL_OLD[ix,iy,iz])*(1.0-relax))   end
    if (ix<=size(η   ,1) && iy<=size(η   ,2) && iz<=size(η   ,3))
        η[ix,iy,iz]    = 2.0/(1.0/η_INI[ix,iy,iz] + 1.0/η_PL[ix,iy,iz])                         end
    return
end

macro dVX(ix,iy,iz)        esc(:(-(   P[$ix+1,$iy+1,$iz+1]-   P[$ix,$iy+1,$iz+1])*_dx +
                                  (τ_XX[$ix+1,$iy+1,$iz+1]-τ_XX[$ix,$iy+1,$iz+1])*_dx +
                                  (τ_XY[$ix  ,$iy+1,$iz  ]-τ_XY[$ix  ,$iy,$iz  ])*_dy +
                                  (τ_XZ[$ix  ,$iy  ,$iz+1]-τ_XZ[$ix  ,$iy  ,$iz])*_dz    ))    end
macro dVY(ix,iy,iz)        esc(:( (τ_XY[$ix+1,$iy  ,$iz  ]-τ_XY[$ix,$iy  ,$iz  ])*_dx -
                                  (   P[$ix+1,$iy+1,$iz+1]-   P[$ix+1,$iy,$iz+1])*_dy +
                                  (τ_YY[$ix+1,$iy+1,$iz+1]-τ_YY[$ix+1,$iy,$iz+1])*_dy +
                                  (τ_YZ[$ix  ,$iy  ,$iz+1]-τ_YZ[$ix  ,$iy  ,$iz])*_dz    ))    end
macro dVZ(ix,iy,iz)        esc(:( (τ_XZ[$ix+1,$iy  ,$iz  ]-τ_XZ[$ix,$iy  ,$iz  ])*_dx +
                                  (τ_YZ[$ix  ,$iy+1,$iz  ]-τ_YZ[$ix  ,$iy,$iz  ])*_dy -
                                  (   P[$ix+1,$iy+1,$iz+1]-   P[$ix+1,$iy+1,$iz])*_dz +
                                  (τ_ZZ[$ix+1,$iy+1,$iz+1]-τ_ZZ[$ix+1,$iy+1,$iz])*_dz -
                                  g*(0.5*(ρ[$ix+1,$iy+1,$iz] + ρ[$ix+1,$iy+1,$iz+1]))    ))    end
macro avxi_DT_R(ix,iy,iz)  esc(:(0.5*(DT_R[$ix  ,$iy+1,$iz+1] + DT_R[$ix+1,$iy+1,$iz+1])    ))    end
macro avyi_DT_R(ix,iy,iz)  esc(:(0.5*(DT_R[$ix+1,$iy  ,$iz+1] + DT_R[$ix+1,$iy+1,$iz+1])    ))    end
macro avzi_DT_R(ix,iy,iz)  esc(:(0.5*(DT_R[$ix+1,$iy+1,$iz  ] + DT_R[$ix+1,$iy+1,$iz+1])    ))    end
@parallel_indices (ix,iy,iz) function compute_V!(  VX::Data.Array ,   VY::Data.Array ,   VZ::Data.Array ,
                                                 τ_XX::Data.Array , τ_YY::Data.Array , τ_ZZ::Data.Array,
                                                 τ_XY::Data.Array , τ_XZ::Data.Array , τ_YZ::Data.Array ,
                                                 DT_R::Data.Array ,    P::Data.Array ,    ρ::Data.Array,
                                                    η::Data.Array ,  ηSM::Data.Array , vpdt::Data.Number,
                                                  dmp::Data.Number,   lx::Data.Number,
                                                  _dx::Data.Number,  _dy::Data.Number,  _dz::Data.Number,
                                                    g::Data.Number)
    if (ix<=size(VX,1)-2 && iy<=size(VX,2)-2 && iz<=size(VX,3)-2)
        VX[ix+1,iy+1,iz+1] = VX[ix+1,iy+1,iz+1] + @dVX(ix,iy,iz) * @avxi_DT_R(ix,iy,iz)    end
    if (ix<=size(VY,1)-2 && iy<=size(VY,2)-2 && iz<=size(VY,3)-2)
        VY[ix+1,iy+1,iz+1] = VY[ix+1,iy+1,iz+1] + @dVY(ix,iy,iz) * @avyi_DT_R(ix,iy,iz)    end
    if (ix<=size(VZ,1)-2 && iy<=size(VZ,2)-2 && iz<=size(VZ,3)-2)
        VZ[ix+1,iy+1,iz+1] = VZ[ix+1,iy+1,iz+1] + @dVZ(ix,iy,iz) * @avzi_DT_R(ix,iy,iz)    end
    return
end

# CHECK ERROR ===============================================================================================
@parallel_indices (ix,iy,iz) function error!(A::Data.Array, B::Data.Array)
    if (ix<=size(A,1) && iy<=size(A,2) && iz<=size(A,3))  A[ix,iy,iz] = abs(A[ix,iy,iz] - B[ix,iy,iz])  end
    return
end

# ===========================================================================================================
@views function CART_3D_Inclusion_Perf()
    num     = 1
    runtime = 0.0
    # physical parameters -----------------------------------------------------------------------------------
    η0       = 1.0                      # Pas   , media viscosity
    vr       = 1e2                      #         viscosity ratio out/in
    ρ0       = 0.0                      # kg/m^3, media density
    ρ_in     = ρ0 - 10.0                # kg/m^3, inclusion density
    g        = 1.0                      # m/s^2 , gravitational acceleration
    n_exp    = 5.0                      #         power-law exponent
    τ_C      = 1.0                      # Pa    , characteristic stress
    relax    = 1e-3                     #         relaxation parameter (power-law)
    lx       = 6.0                      # m     , model dimension in x
    ly       = 1.0*lx                   # m     , model dimension in y
    lz       = 1.0*lx                   # m     , model dimension in z
    radius   = 1.0                      # m     , radius of the inclusion
    sh       = 1.0                      # m/s   , shearing velocity
    # numerics ----------------------------------------------------------------------------------------------
    nx      = 208 - 1                    # number of grid cells in direction x
    ny      = 208 - 1                    # number of grid cells in direction y
    nz      = 208 - 1                    # number of grid cells in direction z
    εnonl   = 5e-7                      # pseudo-transient loop exit criteria
    nt      = 1                         # number of time steps
    maxiter = 1e5                       # maximum number of pseudo-transient iterations
    nout    = 1e2                       # pseudo-transient plotting frequency
    CFL     = 1.0/(2.0 + 4.5*log10(vr)) # Courant-Friedrichs-Lewy condition
    dmp     = 4.5                       # damping parameter
    # preprocessing -----------------------------------------------------------------------------------------
    dx     = lx/(nx-1)                                      # size of cell in direction x
    dy     = ly/(ny-1)                                      # size of cell in direction y
    dz     = lz/(nz-1)                                      # size of cell in direction z
    _dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz                  # 1/size of cells
    X_i    = range(-lx/2.0, lx/2.0, length=nx)              # coordinates of grid points in x
    Y_i    = range(-ly/2.0, ly/2.0, length=ny)              # coordinates of grid points in y
    Z_i    = range(-lz/2.0, lz/2.0, length=nz)              # coordinates of grid points in z
    X_i_vx = range(-(lx+dx)/2.0, (lx+dx)/2.0, length=nx+1)  # coordinates of grid points in r_vr
    Z_i_vz = range(-(lz+dz)/2.0, (lz+dz)/2.0, length=nz+1)  # coordinates of grid points in z_vz
    (X3D ,Y3D ,Z3D ) = ([xc for xc = X_i   ,yc = Y_i,zc = Z_i   ],[yc for xc = X_i   ,yc = Y_i,zc = Z_i   ],
                        [zc for xc = X_i   ,yc = Y_i,zc = Z_i   ])    # grid of coordinates
    (X_VX,Y_VX,Z_VX) = ([xc for xc = X_i_vx,yc = Y_i,zc = Z_i   ],[yc for xc = X_i_vx,yc = Y_i,zc = Z_i   ],
                        [zc for xc = X_i_vx,yc = Y_i,zc = Z_i   ])    # grid of coordinates
    (X_VZ,Y_VZ,Z_VZ) = ([xc for xc = X_i   ,yc = Y_i,zc = Z_i_vz],[yc for xc = X_i   ,yc = Y_i,zc = Z_i_vz],
                        [zc for xc = X_i   ,yc = Y_i,zc = Z_i_vz])    # grid of coordinates
    X3D  = Data.Array(X3D)
    Y3D  = Data.Array(Y3D)
    Z3D  = Data.Array(Z3D)
    X_VX = Data.Array(X_VX)
    Y_VX = Data.Array(Y_VX)
    Z_VX = Data.Array(Z_VX)
    X_VZ = Data.Array(X_VZ)
    Y_VZ = Data.Array(Y_VZ)
    Z_VZ = Data.Array(Z_VZ)
    # initialization and boundary conditions ----------------------------------------------------------------
    print("Starting initialization ... ")
    P         =   @zeros(nx  , ny  , nz  )
    VX        =   @zeros(nx+1, ny  , nz  )
    VY        =   @zeros(nx  , ny+1, nz  )
    VZ        =   @zeros(nx  , ny  , nz+1)
    ERR_VX    =    @ones(nx+1, ny  , nz  )
    ERR_VY    =    @ones(nx  , ny+1, nz  )
    ERR_VZ    =    @ones(nx  , ny  , nz+1)
    τ_XX      =   @zeros(nx  , ny  , nz  )
    τ_YY      =   @zeros(nx  , ny  , nz  )
    τ_ZZ      =   @zeros(nx  , ny  , nz  )
    τ_XY      =   @zeros(nx-1, ny-1, nz-2)
    τ_XZ      =   @zeros(nx-1, ny-2, nz-1)
    τ_YZ      =   @zeros(nx-2, ny-1, nz-1)
    ρ         = ρ0*@ones(nx  , ny  , nz  )
    η         = η0*@ones(nx  , ny  , nz  )
    η_INI     =   @zeros(nx  , ny  , nz  )
    η_PL      =   @zeros(nx  , ny  , nz  )
    η_PL_OLD  =   @zeros(nx  , ny  , nz  )
    ηSM       =   @zeros(nx  , ny  , nz  )
    DT_R      =   @zeros(nx  , ny  , nz  )
    τII       =   @zeros(nx  , ny  , nz  )
    err_evo   = []; iter_evo = []
    iters     = 0.0
    @parallel initialize_inclusion!(ρ, X3D, Y3D, Z3D, radius, ρ_in)
    @parallel initialize_inclusion!(η, X3D, Y3D, Z3D, radius, η0/vr)
    @parallel initialize_velocity!(VX, Y3D, sh)
    vpdt = dx*CFL
    @parallel equal!(η_INI, η)
    @parallel equal!(η_PL , η)
    # action ------------------------------------------------------------------------------------------------
    print("Starting calculation ... \n")
    runtime = @elapsed for it = 1:nt # time loop
        for iter = 1:maxiter # pseudo-transient loop
            # iteration strategy ----------------------------------------------------------------------------
            @parallel equal!(ηSM, η)
            @parallel maxloc!(ηSM, η)
            @parallel copy_boundary!(ηSM)
            @parallel timesteps!(DT_R, η, ηSM, vpdt, dmp, lx)
            # SOLVER ----------------------------------------------------------------------------------------
            @parallel compute_P!(P , τ_XX, τ_YY, τ_ZZ, τ_XY, τ_XZ, τ_YZ, VX, VY, VZ, η, ηSM, dmp, vpdt,
                                 lx, _dx , _dy , _dz )
            if n_exp > 1
                @parallel compute_TII!(τII, τ_XX, τ_YY, τ_ZZ, τ_XY, τ_XZ, τ_YZ)
                @parallel copy_boundary!(τII)
                @parallel equal!(η_PL_OLD, η_PL)
                @parallel power_law!(η_PL, η, τII, η_PL_OLD, η_INI, τ_C, n_exp, relax)
            end
            if mod(iter,nout) == 0
                @parallel equal!(ERR_VX, VX)
                @parallel equal!(ERR_VY, VY)
                @parallel equal!(ERR_VZ, VZ)
            end
            @parallel compute_V!(VX , VY, VZ , τ_XX, τ_YY, τ_ZZ, τ_XY, τ_XZ, τ_YZ, DT_R, P, ρ, η, ηSM, vpdt,
                                 dmp, lx, _dx, _dy , _dz , g)
            @parallel bc_x_lin!(VX,Y_VX,sh)
            @parallel bc_x_0!(VY)
            @parallel bc_x_0!(VZ)
            @parallel bc_y_fact!(VX,Y_VX,sh)
            @parallel bc_y_0!(VY)
            @parallel bc_y_fact!(VZ,Y_VZ,0.0)
            @parallel copy_bc_z!(VX)
            @parallel copy_bc_z!(VY)
            @parallel bc_z_0!(VZ)
            # pseudo-transient loop exit criteria -----------------------------------------------------------
            if mod(iter,nout) == 0
                @parallel error!(ERR_VX, VX)
                @parallel error!(ERR_VY, VY)
                @parallel error!(ERR_VZ, VZ)
                err_vx = maximum(ERR_VX[:])./maximum(abs.(VX[:]))
                err_vy = maximum(ERR_VY[:])./maximum(abs.(VY[:]))
                err_vz = maximum(ERR_VZ[:])./maximum(abs.(VZ[:]))
                err    = max(err_vx, err_vy, err_vz)
                if err<εnonl && iter>20
                    iters = iter
                    break
                end
                # postprocessing
                push!(err_evo, err)
                push!(iter_evo,iter)
                @printf("iter %d, err=%1.3e \n", iter, err)
            end
        end
    end
    # SAVING ------------------------------------------------------------------------------------------------
    if saveflag
        !ispath("../out_cart") && mkdir("../out_cart")
        err_evo  = Data.Array(err_evo)
        iter_evo = Data.Array(iter_evo)
        Save_phys(num, η0, ρ0, ρ_in, n_exp, τ_C, vr, g, sh, radius, dmp)
        Save_infos(num, lx, ly, lz, nx, ny, nz, εnonl, runtime)
        SaveArray("X3D"     , X3D     )
        SaveArray("Y3D"     , Y3D     )
        SaveArray("Z3D"     , Z3D     )
        SaveArray("ETAS"    , η       )
        SaveArray("RHO"     , ρ       )
        SaveArray("P"       , P       )
        SaveArray("VX"      , VX      )
        SaveArray("VY"      , VY      )
        SaveArray("VZ"      , VZ      )
        SaveArray("TAU_XX"  , τ_XX    )
        SaveArray("TAU_YY"  , τ_YY    )
        SaveArray("TAU_ZZ"  , τ_ZZ    )
        SaveArray("TAU_XY"  , τ_XY    )
        SaveArray("TAU_XZ"  , τ_XZ    )
        SaveArray("TAU_YZ"  , τ_YZ    )
        SaveArray("TII"     , τII     )
        # SaveArray("iters"   , iters   )
        SaveArray("err_evo" , err_evo )
        SaveArray("iter_evo", iter_evo)
    end
    return
end

@time CART_3D_Inclusion_Perf()
