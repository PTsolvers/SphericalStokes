const USE_GPU   = haskey(ENV, "USE_GPU")   ? parse(Bool, ENV["USE_GPU"]  ) : true
const do_save   = haskey(ENV, "DO_SAVE")   ? parse(Bool, ENV["DO_SAVE"]  ) : false
const do_save_p = haskey(ENV, "DO_SAVE_P") ? parse(Bool, ENV["DO_SAVE_P"]) : false
const nr        = haskey(ENV, "NR"     )   ? parse(Int , ENV["NR"]       ) : 16*8 - 1
const nθ        = haskey(ENV, "NTH"    )   ? parse(Int , ENV["NTH"]      ) : 16*8 - 1
const nφ        = haskey(ENV, "NPH"    )   ? parse(Int , ENV["NPH"]      ) : 16*8 - 1

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid, Printf, Statistics, LinearAlgebra
import MPI
using HDF5, LightXML

include(joinpath(@__DIR__,"data_io2.jl"))

@views inn(A)    =  A[2:end-1,2:end-1,2:end-1]
@views av_x2i(A) = (A[3:end-1,2:end-1,2:end-1].+A[2:end-2,2:end-1,2:end-1]).*0.5
@views av_y2i(A) = (A[2:end-1,3:end-1,2:end-1].+A[2:end-1,2:end-2,2:end-1]).*0.5
@views av_z2i(A) = (A[2:end-1,2:end-1,3:end-1].+A[2:end-1,2:end-1,2:end-2]).*0.5
@views av_xya(A) = (A[1:end-1,1:end-1,:].+A[2:end,1:end-1,:].+A[1:end-1,2:end,:].+A[2:end,2:end,:]).*0.25
@views av_xza(A) = (A[1:end-1,:,1:end-1].+A[2:end,:,1:end-1].+A[1:end-1,:,2:end].+A[2:end,:,2:end]).*0.25
@views av_yza(A) = (A[:,1:end-1,1:end-1].+A[:,2:end,1:end-1].+A[:,1:end-1,2:end].+A[:,2:end,2:end]).*0.25
@views     av(A) = (A[1:end-1,1:end-1,1:end-1].+A[2:end,1:end-1,1:end-1].+A[1:end-1,2:end,1:end-1].+A[1:end-1,1:end-1,2:end].+A[2:end,2:end,1:end-1].+A[2:end,1:end-1,2:end].+A[1:end-1,2:end,2:end].+A[2:end,2:end,2:end]).*0.125

max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

@parallel function sph2cart!(X::Data.Array, Y::Data.Array, Z::Data.Array, R::Data.Array, θ::Data.Array, φ::Data.Array)

    @all(X) = @all(R) * cos(π/2 - @all(θ)) * cos(@all(φ)) # ATTENTION: `π/2 - ` is specific to this configuration
    @all(Y) = @all(R) * sin(π/2 - @all(θ)) * cos(@all(φ)) # ATTENTION: `π/2 - ` is specific to this configuration
    @all(Z) = @all(R)                      * sin(@all(φ))
    return
end

@parallel_indices (ix,iy,iz) function equal3!(VR_err::Data.Array, Vθ_err::Data.Array, Vφ_err::Data.Array,
                                                  VR::Data.Array,     Vθ::Data.Array,     Vφ::Data.Array)

    if (ix<=size(VR_err,1) && iy<=size(VR_err,2) && iz<=size(VR_err,3))
        VR_err[ix,iy,iz] = VR[ix,iy,iz]
    end
    if (ix<=size(Vθ_err,1) && iy<=size(Vθ_err,2) && iz<=size(Vθ_err,3))
        Vθ_err[ix,iy,iz] = Vθ[ix,iy,iz]
    end
    if (ix<=size(Vφ_err,1) && iy<=size(Vφ_err,2) && iz<=size(Vφ_err,3))
        Vφ_err[ix,iy,iz] = Vφ[ix,iy,iz]
    end
    return
end

@parallel_indices (iy,iz) function copy_bc_r!(A::Data.Array)

    A[1  ,iy,iz] = A[2    ,iy,iz]
    A[end,iy,iz] = A[end-1,iy,iz]
    return
end

@parallel_indices (ix,iz) function copy_bc_θ!(A::Data.Array)

    A[ix,1  ,iz] = A[ix,2    ,iz]
    A[ix,end,iz] = A[ix,end-1,iz]
    return
end

@parallel_indices (ix,iy) function copy_bc_φ!(A::Data.Array)

    A[ix,iy,1  ] = A[ix,iy,2    ]
    A[ix,iy,end] = A[ix,iy,end-1]
    return
end

@parallel_indices (iy,iz) function bc_mx!(A::Data.Array)

    A[1  ,iy,iz] = -A[2    ,iy,iz]
    A[end,iy,iz] = -A[end-1,iy,iz]
    return
end

@parallel_indices (ix,iz) function bc_my!(A::Data.Array)

    A[ix,1  ,iz] = -A[ix,2    ,iz]
    A[ix,end,iz] = -A[ix,end-1,iz]
    return
end

@parallel_indices (ix,iz) function bc_0y!(A::Data.Array)

    A[ix,1  ,iz] = 0.0
    A[ix,end,iz] = 0.0
    return
end

@parallel_indices (ix,iy) function bc_0z!(A::Data.Array)

    A[ix,iy,1  ] = 0.0
    A[ix,iy,end] = 0.0
    return
end

@parallel_indices (ix,iz) function bc_1!(A::Data.Array, B::Data.Array, c::Data.Number, d::Data.Number)

    A[ix,1  ,iz] = (B[ix,1  ,iz] - π/2.0)c*d
    A[ix,end,iz] = (B[ix,end,iz] - π/2.0)c*d
    return
end

@parallel_indices (ix) function bc_2!(A::Data.Array, B::Data.Array, c::Data.Number, d::Data.Number)

    A[ix,1  ,end] = (B[ix,1  ,end-1] - π/2.0)c*d
    A[ix,end,end] = (B[ix,end,end-1] - π/2.0)c*d
    return
end

@parallel_indices (ix,iy) function bc_3!(A::Data.Array, B::Data.Array, c::Data.Number, d::Data.Number)

    A[ix,iy,1  ]  = 2.0*(B[ix,iy,1    ] - π/2.0)*c*d - A[ix,iy,2    ]
    A[ix,iy,end]  = 2.0*(B[ix,iy,end-1] - π/2.0)*c*d - A[ix,iy,end-1]
    return
end

@parallel_indices (ix,iy,iz) function initialize_inclusion!(A::Data.Array, R::Data.Array, θ::Data.Array, φ::Data.Array,
                                                            r::Data.Number, radius::Data.Number, in::Data.Number)
    if (ix<=size(A,1) && iy<=size(A,2) && iz<=size(A,3))
        if (((R[ix,iy,iz]-r)^2 + ((θ[ix,iy,iz]-pi/2.0)*r)^2 + (φ[ix,iy,iz]*r)^2) < radius)  A[ix,iy,iz] = in    end
    end
    return
end

@parallel_indices (ix,iy,iz) function initialize_velocity!(V::Data.Array, COORD::Data.Array, fact::Data.Number, r::Data.Number)

    if (ix<=size(V,1) && iy<=size(V,2) && iz<=size(V,3)-1)  V[ix,iy,iz] = (COORD[ix,iy,iz  ] - pi/2.0)*r*fact   end
    if (ix<=size(V,1) && iy<=size(V,2) && iz==size(V,3)  )  V[ix,iy,iz] = (COORD[ix,iy,iz-1] - pi/2.0)*r*fact   end
    return
end

@parallel function maxloc!(ηSM::Data.Array, η::Data.Array)

    @inn(ηSM) = @maxloc(η)
    return
end

macro KBDT(ix,iy,iz)        esc(:(  dmp * 2.0 * pi * vpdt / lr * ηSM[$ix,$iy,$iz]   ))  end
macro GSDT(ix,iy,iz)        esc(:(        4.0 * pi * vpdt / lr * ηSM[$ix,$iy,$iz]   ))  end
macro avxa_VR(ix,iy,iz)     esc(:(  ((   VR[$ix  ,$iy  ,$iz  ] +    VR[$ix+1,$iy  ,$iz  ])*0.5) ))  end
macro avxi_Vθ(ix,iy,iz)     esc(:(  ((   Vθ[$ix  ,$iy+1,$iz+1] +    Vθ[$ix+1,$iy+1,$iz+1])*0.5) ))  end
macro avya_Vθ(ix,iy,iz)     esc(:(  ((   Vθ[$ix  ,$iy  ,$iz  ] +    Vθ[$ix  ,$iy+1,$iz  ])*0.5) ))  end
macro avxi_Vφ(ix,iy,iz)     esc(:(  ((   Vφ[$ix  ,$iy+1,$iz+1] +    Vφ[$ix+1,$iy+1,$iz+1])*0.5) ))  end
macro avyi_Vφ(ix,iy,iz)     esc(:(  ((   Vφ[$ix+1,$iy  ,$iz+1] +    Vφ[$ix+1,$iy+1,$iz+1])*0.5) ))  end
macro avxyi_R(ix,iy,iz)     esc(:(  ((    R[$ix  ,$iy  ,$iz+1] +     R[$ix  ,$iy+1,$iz+1] +     R[$ix+1,$iy  ,$iz+1] +     R[$ix+1,$iy+1,$iz+1])*0.25)  ))  end
macro avxzi_R(ix,iy,iz)     esc(:(  ((    R[$ix  ,$iy+1,$iz  ] +     R[$ix  ,$iy+1,$iz+1] +     R[$ix+1,$iy+1,$iz  ] +     R[$ix+1,$iy+1,$iz+1])*0.25)  ))  end
macro avyzi_R(ix,iy,iz)     esc(:(  ((    R[$ix+1,$iy  ,$iz  ] +     R[$ix+1,$iy  ,$iz+1] +     R[$ix+1,$iy+1,$iz  ] +     R[$ix+1,$iy+1,$iz+1])*0.25)  ))  end
macro avxzi_θ(ix,iy,iz)     esc(:(  ((    θ[$ix  ,$iy+1,$iz  ] +     θ[$ix  ,$iy+1,$iz+1] +     θ[$ix+1,$iy+1,$iz  ] +     θ[$ix+1,$iy+1,$iz+1])*0.25)  ))  end
macro avyzi_θ(ix,iy,iz)     esc(:(  ((    θ[$ix+1,$iy  ,$iz  ] +     θ[$ix+1,$iy  ,$iz+1] +     θ[$ix+1,$iy+1,$iz  ] +     θ[$ix+1,$iy+1,$iz+1])*0.25)  ))  end
macro avxyi_GSDT(ix,iy,iz)  esc(:(  ((@GSDT($ix  ,$iy  ,$iz+1) + @GSDT($ix  ,$iy+1,$iz+1) + @GSDT($ix+1,$iy  ,$iz+1) + @GSDT($ix+1,$iy+1,$iz+1))*0.25)  ))  end
macro avxzi_GSDT(ix,iy,iz)  esc(:(  ((@GSDT($ix  ,$iy+1,$iz  ) + @GSDT($ix  ,$iy+1,$iz+1) + @GSDT($ix+1,$iy+1,$iz  ) + @GSDT($ix+1,$iy+1,$iz+1))*0.25)  ))  end
macro avyzi_GSDT(ix,iy,iz)  esc(:(  ((@GSDT($ix+1,$iy  ,$iz  ) + @GSDT($ix+1,$iy  ,$iz+1) + @GSDT($ix+1,$iy+1,$iz  ) + @GSDT($ix+1,$iy+1,$iz+1))*0.25)  ))  end
macro avxyi_η(ix,iy,iz)     esc(:(  ((    η[$ix  ,$iy  ,$iz+1] +     η[$ix  ,$iy+1,$iz+1] +     η[$ix+1,$iy  ,$iz+1] +     η[$ix+1,$iy+1,$iz+1])*0.25)  ))  end
macro avxzi_η(ix,iy,iz)     esc(:(  ((    η[$ix  ,$iy+1,$iz  ] +     η[$ix  ,$iy+1,$iz+1] +     η[$ix+1,$iy+1,$iz  ] +     η[$ix+1,$iy+1,$iz+1])*0.25)  ))  end
macro avyzi_η(ix,iy,iz)     esc(:(  ((    η[$ix+1,$iy  ,$iz  ] +     η[$ix+1,$iy  ,$iz+1] +     η[$ix+1,$iy+1,$iz  ] +     η[$ix+1,$iy+1,$iz+1])*0.25)  ))  end
macro DIVV(ix,iy,iz)        esc(:((                                         VR[$ix+1,$iy,$iz] - VR[$ix,$iy,$iz])*_dr +
                                    1.0/R[$ix,$iy,$iz]                    *(Vθ[$ix,$iy+1,$iz] - Vθ[$ix,$iy,$iz])*_dθ +
                                    1.0/R[$ix,$iy,$iz]/sin(θ[$ix,$iy,$iz])*(Vφ[$ix,$iy,$iz+1] - Vφ[$ix,$iy,$iz])*_dφ +
                                    2.0*@avxa_VR(ix,iy,iz)/R[$ix,$iy,$iz] + @avya_Vθ(ix,iy,iz)/R[$ix,$iy,$iz]*cot(θ[$ix,$iy,$iz])   ))  end
macro DRR(ix,iy,iz)         esc(:(  (VR[$ix+1,$iy,$iz] - VR[$ix,$iy,$iz])*_dr - 1.0/3.0*@DIVV(ix,iy,iz) ))  end
macro Dθθ(ix,iy,iz)         esc(:(  1.0/R[$ix,$iy,$iz]*(Vθ[$ix,$iy+1,$iz] - Vθ[$ix,$iy,$iz])*_dθ +
                                    @avxa_VR(ix,iy,iz)/R[$ix,$iy,$iz] - 1.0/3.0*@DIVV(ix,iy,iz) ))  end
macro Dφφ(ix,iy,iz)         esc(:(  1.0/R[$ix,$iy,$iz]/sin(θ[$ix,$iy,$iz])*(Vφ[$ix,$iy,$iz+1] - Vφ[$ix,$iy,$iz])*_dφ +
                                    @avxa_VR(ix,iy,iz)/R[$ix,$iy,$iz] + @avya_Vθ($ix,$iy,$iz)/R[$ix,$iy,$iz]*cot(θ[$ix,$iy,$iz]) - 1.0/3.0*@DIVV(ix,iy,iz)  ))  end
macro DRθ(ix,iy,iz)         esc(:(                         (Vθ[$ix+1,$iy+1,$iz+1] - Vθ[$ix,$iy+1,$iz+1])*_dr +
                                    1.0/@avxyi_R(ix,iy,iz)*(VR[$ix+1,$iy+1,$iz+1] - VR[$ix+1,$iy,$iz+1])*_dθ -
                                    @avxi_Vθ(ix,iy,iz)/@avxyi_R(ix,iy,iz)   ))  end
macro DRφ(ix,iy,iz)         esc(:(                                                 (Vφ[$ix+1,$iy+1,$iz+1] - Vφ[$ix,$iy+1,$iz+1])*_dr +
                                    1.0/@avxzi_R(ix,iy,iz)/sin(@avxzi_θ(ix,iy,iz))*(VR[$ix+1,$iy+1,$iz+1] - VR[$ix+1,$iy+1,$iz])*_dφ -
                                    @avxi_Vφ(ix,iy,iz)/@avxzi_R(ix,iy,iz)   ))  end
macro Dθφ(ix,iy,iz)         esc(:(  1.0/@avyzi_R(ix,iy,iz)                        *(Vφ[$ix+1,$iy+1,$iz+1] - Vφ[$ix+1,$iy,$iz+1])*_dθ +
                                    1.0/@avyzi_R(ix,iy,iz)/sin(@avyzi_θ(ix,iy,iz))*(Vθ[$ix+1,$iy+1,$iz+1] - Vθ[$ix+1,$iy+1,$iz])*_dφ -
                                    @avyi_Vφ(ix,iy,iz)/@avyzi_R(ix,iy,iz)*cot(@avyzi_θ(ix,iy,iz))  ))  end

@parallel_indices (ix,iy,iz) function compute_P!(   P::Data.Array , DT_R::Data.Array ,
                                                 τ_RR::Data.Array , τ_θθ::Data.Array , τ_φφ::Data.Array ,
                                                 τ_Rθ::Data.Array , τ_Rφ::Data.Array , τ_θφ::Data.Array ,
                                                   VR::Data.Array ,   Vθ::Data.Array ,   Vφ::Data.Array ,
                                                    R::Data.Array ,    θ::Data.Array ,    η::Data.Array , ηSM::Data.Array,
                                                  _dr::Data.Number,  _dθ::Data.Number,  _dφ::Data.Number,
                                                  dmp::Data.Number, vpdt::Data.Number,   lr::Data.Number)

    if (ix<=size(P,1) && iy<=size(P,2) && iz<=size(P,3))
        P[ix,iy,iz] = P[ix,iy,iz] - @KBDT(ix,iy,iz) * @DIVV(ix,iy,iz)
        # @all(P)    = @all(P) - @all(KBDT * (@all(DIVV) + (@all(P) - @all(P_OLD))/dt * beta + @all(P)/eta0)

        τ_RR[ix,iy,iz] = (τ_RR[ix,iy,iz] + @GSDT(ix,iy,iz)*2.0*@DRR(ix,iy,iz))/(1.0 + @GSDT(ix,iy,iz)/η[ix,iy,iz])
        τ_θθ[ix,iy,iz] = (τ_θθ[ix,iy,iz] + @GSDT(ix,iy,iz)*2.0*@Dθθ(ix,iy,iz))/(1.0 + @GSDT(ix,iy,iz)/η[ix,iy,iz])
        τ_φφ[ix,iy,iz] = (τ_φφ[ix,iy,iz] + @GSDT(ix,iy,iz)*2.0*@Dφφ(ix,iy,iz))/(1.0 + @GSDT(ix,iy,iz)/η[ix,iy,iz])

        DT_R[ix,iy,iz] = vpdt^2 / (@KBDT(ix,iy,iz) + @GSDT(ix,iy,iz)/(1.0 + @GSDT(ix,iy,iz)/η[ix,iy,iz]))
    end

    if (ix<=size(τ_Rθ,1) && iy<=size(τ_Rθ,2) && iz<=size(τ_Rθ,3))
        τ_Rθ[ix,iy,iz] = (τ_Rθ[ix,iy,iz] + @avxyi_GSDT(ix,iy,iz)*@DRθ(ix,iy,iz))/(1.0 + @avxyi_GSDT(ix,iy,iz)/@avxyi_η(ix,iy,iz))
    end

    if (ix<=size(τ_Rφ,1) && iy<=size(τ_Rφ,2) && iz<=size(τ_Rφ,3))
        τ_Rφ[ix,iy,iz] = (τ_Rφ[ix,iy,iz] + @avxzi_GSDT(ix,iy,iz)*@DRφ(ix,iy,iz))/(1.0 + @avxzi_GSDT(ix,iy,iz)/@avxzi_η(ix,iy,iz))
    end

    if (ix<=size(τ_θφ,1) && iy<=size(τ_θφ,2) && iz<=size(τ_θφ,3))
        τ_θφ[ix,iy,iz] = (τ_θφ[ix,iy,iz] + @avyzi_GSDT(ix,iy,iz)*@Dθφ(ix,iy,iz))/(1.0 + @avyzi_GSDT(ix,iy,iz)/@avyzi_η(ix,iy,iz))
    end
    return
end

@parallel_indices (ix,iy,iz) function compute_TII!( τII::Data.Array, τ_RR::Data.Array, τ_θθ::Data.Array, τ_φφ::Data.Array,
                                                   τ_Rθ::Data.Array, τ_Rφ::Data.Array, τ_θφ::Data.Array)

    if (ix<=size(τII,1)-2 && iy<=size(τII,2)-2 && iz<=size(τII,3)-2)
        τII[ix+1,iy+1,iz+1] = sqrt(1.0/2.0 * (τ_RR[ix+1,iy+1,iz+1]^2.0 + τ_θθ[ix+1,iy+1,iz+1]^2.0 + τ_φφ[ix+1,iy+1,iz+1]^2.0) +
                              ((τ_Rθ[ix,iy,iz] + τ_Rθ[ix,iy+1,iz  ] + τ_Rθ[ix+1,iy  ,iz] + τ_Rθ[ix+1,iy+1,iz  ])*0.25)^2.0 +
                              ((τ_Rφ[ix,iy,iz] + τ_Rφ[ix,iy  ,iz+1] + τ_Rφ[ix+1,iy  ,iz] + τ_Rφ[ix+1,iy  ,iz+1])*0.25)^2.0 +
                              ((τ_θφ[ix,iy,iz] + τ_θφ[ix,iy  ,iz+1] + τ_θφ[ix  ,iy+1,iz] + τ_θφ[ix  ,iy+1,iz+1])*0.25)^2.0)
    end
    return
end

@parallel_indices (ix,iy,iz) function power_law!(    η::Data.Array ,   τII::Data.Array , η_ini::Data.Array,
                                                 s_ref::Data.Number, n_exp::Data.Number)

    if (ix<=size(η,1) && iy<=size(η,2) && iz<=size(η,3))
        η[ix,iy,iz] = 2.0/(1.0/η_ini[ix,iy,iz] + 1.0/(η_ini[ix,iy,iz] * (τII[ix,iy,iz]/s_ref)^(1.0-n_exp)))
    end
    return
end

macro σRR(ix,iy,iz)         esc(:(  -P[$ix,$iy,$iz] + τ_RR[$ix,$iy,$iz] ))  end
macro σθθ(ix,iy,iz)         esc(:(  -P[$ix,$iy,$iz] + τ_θθ[$ix,$iy,$iz] ))  end
macro σφφ(ix,iy,iz)         esc(:(  -P[$ix,$iy,$iz] + τ_φφ[$ix,$iy,$iz] ))  end
macro ρG(ix,iy,iz)          esc(:(   ρ[$ix,$iy,$iz]*g   ))  end
macro avxi_R(ix,iy,iz)      esc(:(  ((   R[$ix  ,$iy+1,$iz+1] +    R[$ix+1,$iy+1,$iz+1])*0.5)   ))  end
macro avyi_R(ix,iy,iz)      esc(:(  ((   R[$ix+1,$iy  ,$iz+1] +    R[$ix+1,$iy+1,$iz+1])*0.5)   ))  end
macro avzi_R(ix,iy,iz)      esc(:(  ((   R[$ix+1,$iy+1,$iz  ] +    R[$ix+1,$iy+1,$iz+1])*0.5)   ))  end
macro avxi_θ(ix,iy,iz)      esc(:(  ((   θ[$ix  ,$iy+1,$iz+1] +    θ[$ix+1,$iy+1,$iz+1])*0.5)   ))  end
macro avyi_θ(ix,iy,iz)      esc(:(  ((   θ[$ix+1,$iy  ,$iz+1] +    θ[$ix+1,$iy+1,$iz+1])*0.5)   ))  end
macro avzi_θ(ix,iy,iz)      esc(:(  ((   θ[$ix+1,$iy+1,$iz  ] +    θ[$ix+1,$iy+1,$iz+1])*0.5)   ))  end
macro avxi_σRR(ix,iy,iz)    esc(:(  ((@σRR( ix  , iy+1, iz+1) + @σRR( ix+1, iy+1, iz+1))*0.5)   ))  end
macro avxi_σθθ(ix,iy,iz)    esc(:(  ((@σθθ( ix  , iy+1, iz+1) + @σθθ( ix+1, iy+1, iz+1))*0.5)   ))  end
macro avyi_σθθ(ix,iy,iz)    esc(:(  ((@σθθ( ix+1, iy  , iz+1) + @σθθ( ix+1, iy+1, iz+1))*0.5)   ))  end
macro avxi_σφφ(ix,iy,iz)    esc(:(  ((@σφφ( ix  , iy+1, iz+1) + @σφφ( ix+1, iy+1, iz+1))*0.5)   ))  end
macro avyi_σφφ(ix,iy,iz)    esc(:(  ((@σφφ( ix+1, iy  , iz+1) + @σφφ( ix+1, iy+1, iz+1))*0.5)   ))  end
macro avxa_τRθ(ix,iy,iz)    esc(:(  ((τ_Rθ[$ix  ,$iy  ,$iz  ] + τ_Rθ[$ix+1,$iy  ,$iz  ])*0.5)   ))  end
macro avya_τRθ(ix,iy,iz)    esc(:(  ((τ_Rθ[$ix  ,$iy  ,$iz  ] + τ_Rθ[$ix  ,$iy+1,$iz  ])*0.5)   ))  end
macro avxa_τRφ(ix,iy,iz)    esc(:(  ((τ_Rφ[$ix  ,$iy  ,$iz  ] + τ_Rφ[$ix+1,$iy  ,$iz  ])*0.5)   ))  end
macro avya_τθφ(ix,iy,iz)    esc(:(  ((τ_θφ[$ix  ,$iy  ,$iz  ] + τ_θφ[$ix  ,$iy+1,$iz  ])*0.5)   ))  end
macro avxi_ρG(ix,iy,iz)     esc(:(  (( @ρG( ix  , iy+1, iz+1) +  @ρG( ix+1, iy+1, iz+1))*0.5)   ))  end
macro dVR(ix,iy,iz)         esc(:(                                               (@σRR(ix+1,iy+1,iz+1) - @σRR(ix,iy+1,iz+1))*_dr +
                                    1.0/@avxi_R(ix,iy,iz)                       *(τ_Rθ[ix  ,iy+1,iz  ] - τ_Rθ[ix,iy  ,iz  ])*_dθ +
                                    1.0/@avxi_R(ix,iy,iz)/sin(@avxi_θ(ix,iy,iz))*(τ_Rφ[ix  ,iy  ,iz+1] - τ_Rφ[ix,iy  ,iz  ])*_dφ +
                                    2.0*@avxi_σRR(ix,iy,iz)                     /@avxi_R(ix,iy,iz) -
                                       (@avxi_σθθ(ix,iy,iz)+@avxi_σφφ(ix,iy,iz))/@avxi_R(ix,iy,iz) +
                                        @avya_τRθ(ix,iy,iz)                     /@avxi_R(ix,iy,iz)*cot(@avxi_θ(ix,iy,iz)) -
                                        @avxi_ρG(ix,iy,iz)  ))  end
macro dVθ(ix,iy,iz)         esc(:(                                               (    τ_Rθ[ix+1,iy  ,iz  ] - τ_Rθ[ix  ,iy,iz  ])*_dr +
                                    1.0/@avyi_R(ix,iy,iz)                       *(    @σθθ(ix+1,iy+1,iz+1) - @σθθ(ix+1,iy,iz+1))*_dθ +
                                    1.0/@avyi_R(ix,iy,iz)/sin(@avyi_θ(ix,iy,iz))*(    τ_θφ[ix  ,iy  ,iz+1] - τ_θφ[ix  ,iy,iz  ])*_dφ +
                                    3.0*@avxa_τRθ(ix,iy,iz)                       /@avyi_R(ix  ,iy  ,iz  ) +
                                       (@avyi_σθθ(ix,iy,iz) - @avyi_σφφ(ix,iy,iz))/@avyi_R(ix  ,iy  ,iz  )*cot(@avyi_θ(ix,iy,iz)) ))  end
macro dVφ(ix,iy,iz)         esc(:(                                               (τ_Rφ[ix+1,iy  ,iz  ] - τ_Rφ[ix  ,iy  ,iz])*_dr +
                                    1.0/@avzi_R(ix,iy,iz)                       *(τ_θφ[ix  ,iy+1,iz  ] - τ_θφ[ix  ,iy  ,iz])*_dθ +
                                    1.0/@avzi_R(ix,iy,iz)/sin(@avzi_θ(ix,iy,iz))*(@σφφ(ix+1,iy+1,iz+1) - @σφφ(ix+1,iy+1,iz))*_dφ +
                                    3.0*@avxa_τRφ(ix,iy,iz)/@avzi_R(ix,iy,iz) +
                                    2.0*@avya_τθφ(ix,iy,iz)/@avzi_R(ix,iy,iz)*cot(@avzi_θ(ix,iy,iz))    ))  end

@parallel_indices (ix,iy,iz) function compute_V_ηSM!(  VR::Data.Array ,   Vθ::Data.Array ,   Vφ::Data.Array , DT_R::Data.Array ,
                                                      ηSM::Data.Array ,    η::Data.Array ,
                                                     τ_RR::Data.Array , τ_θθ::Data.Array , τ_φφ::Data.Array ,
                                                     τ_Rθ::Data.Array , τ_Rφ::Data.Array , τ_θφ::Data.Array ,
                                                        P::Data.Array ,    R::Data.Array ,    θ::Data.Array ,    ρ::Data.Array ,
                                                      _dr::Data.Number,  _dθ::Data.Number,  _dφ::Data.Number,    g::Data.Number)

    if (ix<=size(VR,1)-2 && iy<=size(VR,2)-2 && iz<=size(VR,3)-2)
        VR[ix+1,iy+1,iz+1] = VR[ix+1,iy+1,iz+1] + @dVR(ix,iy,iz)*((DT_R[ix,iy+1,iz+1] + DT_R[ix+1,iy+1,iz+1])*0.5)
    end

    if (ix<=size(Vθ,1)-2 && iy<=size(Vθ,2)-2 && iz<=size(Vθ,3)-2)
        Vθ[ix+1,iy+1,iz+1] = Vθ[ix+1,iy+1,iz+1] + @dVθ(ix,iy,iz)*((DT_R[ix+1,iy,iz+1] + DT_R[ix+1,iy+1,iz+1])*0.5)
    end

    if (ix<=size(Vφ,1)-2 && iy<=size(Vφ,2)-2 && iz<=size(Vφ,3)-2)
        Vφ[ix+1,iy+1,iz+1] = Vφ[ix+1,iy+1,iz+1] + @dVφ(ix,iy,iz)*((DT_R[ix+1,iy+1,iz] + DT_R[ix+1,iy+1,iz+1])*0.5)
    end

    if (ix<=size(η,1)-2 && iy<=size(η,2)-2 && iz<=size(η,3)-2)
        ηSM[ix+1,iy+1,iz+1] = max( max( max( max(η[ix+1-1,iy+1  ,iz+1  ], η[ix+1+1,iy+1  ,iz+1  ])  , η[ix+1  ,iy+1  ,iz+1  ] ),
                                             max(η[ix+1  ,iy+1-1,iz+1  ], η[ix+1  ,iy+1+1,iz+1  ]) ),
                                             max(η[ix+1  ,iy+1  ,iz+1-1], η[ix+1  ,iy+1  ,iz+1+1]) )
    end
    return
end

@parallel_indices (ix,iy,iz) function check_err!(VR_err::Data.Array, Vθ_err::Data.Array, Vφ_err::Data.Array,
                                                     VR::Data.Array,     Vθ::Data.Array,     Vφ::Data.Array)

    if (ix<=size(VR,1) && iy<=size(VR,2) && iz<=size(VR,3))
        VR_err[ix,iy,iz] = VR[ix,iy,iz] - VR_err[ix,iy,iz]
    end
    if (ix<=size(Vθ,1) && iy<=size(Vθ,2) && iz<=size(Vθ,3))
        Vθ_err[ix,iy,iz] = Vθ[ix,iy,iz] - Vθ_err[ix,iy,iz]
    end
    if (ix<=size(Vφ,1) && iy<=size(Vφ,2) && iz<=size(Vφ,3))
        Vφ_err[ix,iy,iz] = Vφ[ix,iy,iz] - Vφ_err[ix,iy,iz]
    end
    return
end

# ==============================================================================
@views function SphericalStokes()
    out_path = "../out_visu"
    out_name = "results3D"
    # physical parameters
    η0       = 1.0              # Pa*s  , media viscosity
    vr       = 1e2              #         viscosity ratio out/in
    ρ0       = 0.0              # kg/m^3, media density
    ρ_in     = ρ0 - 10.0        # kg/m^3, inclusion density
    g        = 1.0              # m/s^2 , gravity acceleration
    n_exp    = 5.0              #         power law exponent
    s_ref    = 1.0              # Pa    , reference stress (power law)
    r        = 1000.0           # m     , radius of the total sphere
    lr       = 6.0              # m     , model dimension in r
    lθ       = lr/r             # m     , model dimension in θ
    lφ       = lr/r             # m     , model dimension in z
    radius   = 1.0              # m     , radius of the inclusion
    sh       = 1.0              # m/s   , shearing velocity
    # numerics
    me,dims,nprocs,coords,comm_cart = init_global_grid(nr,nθ,nφ)
    info     = MPI.Info()
    b_width  = (8,4,4)               # boundary width
    εnonl    = 5e-7                  # pseudo-transient loop exit criteria
    nt       = 1                     # number of time steps
    maxiter  = 1e2                   # maximum number of pseudo-transient iterations
    nout     = 2e3                   # pseudo-transient plotting frequency
    CFL      = 1.0/(2.0 + 4.5*log10(vr))
    dmp      = 4.5
    # preprocessing
    nr_g, nθ_g, nφ_g = nx_g(), ny_g(), nz_g()
    dr, dθ, dφ       = lr/(nr_g-1), lθ/(nθ_g-1), lφ/(nφ_g-1)
    _dr, _dθ, _dφ    = 1.0/dr, 1.0/dθ, 1.0/dφ
    P        = @zeros(nr, nθ, nφ)
    R        = Data.Array([r     + x_g(ir,dr,P) - lr/2.0 for ir=1:size(P,1), iθ=1:size(P,2), iφ=1:size(P,3)])
    θ        = Data.Array([π/2.0 + y_g(iθ,dθ,P) - lθ/2.0 for ir=1:size(P,1), iθ=1:size(P,2), iφ=1:size(P,3)])
    φ        = Data.Array([        z_g(iφ,dφ,P) - lφ/2.0 for ir=1:size(P,1), iθ=1:size(P,2), iφ=1:size(P,3)])
    me==0 && print("Starting initialization ... ")
    # initial and boundary conditions
    VR       =   @zeros(nr+1, nθ  , nφ  )
    Vθ       =   @zeros(nr  , nθ+1, nφ  )
    Vφ       =   @zeros(nr  , nθ  , nφ+1)
    VR_err   =   @zeros(nr+1, nθ  , nφ  )
    Vθ_err   =   @zeros(nr  , nθ+1, nφ  )
    Vφ_err   =   @zeros(nr  , nθ  , nφ+1)
    τ_RR     =   @zeros(nr  , nθ  , nφ  )
    τ_θθ     =   @zeros(nr  , nθ  , nφ  )
    τ_φφ     =   @zeros(nr  , nθ  , nφ  )
    τ_Rθ     =   @zeros(nr-1, nθ-1, nφ-2)
    τ_Rφ     =   @zeros(nr-1, nθ-2, nφ-1)
    τ_θφ     =   @zeros(nr-2, nθ-1, nφ-1)
    ρ        = ρ0*@ones(nr  , nθ  , nφ  )
    η        = η0*@ones(nr  , nθ  , nφ  )
    η_ini    =   @zeros(nr  , nθ  , nφ  )
    ηSM      =   @zeros(nr  , nθ  , nφ  )
    DT_R     =   @zeros(nr  , nθ  , nφ  )
    τII      =   @zeros(nr  , nθ  , nφ  )
    X        =   @zeros(nr  , nθ  , nφ  ) # cart coords
    Y        =   @zeros(nr  , nθ  , nφ  ) # cart coords
    Z        =   @zeros(nr  , nθ  , nφ  ) # cart coords
    vpdt     = dr*CFL
    @parallel initialize_inclusion!(ρ, R, θ, φ, r, radius, ρ_in)
    @parallel initialize_inclusion!(η, R, θ, φ, r, radius, η0/vr)
    @parallel initialize_velocity!(Vφ, θ, sh, r)
    update_halo!(ρ, η, Vφ)
    η_ini   .= η
    ηSM     .= η
    me==0 && println("done.")
    do_save && me==0 && !ispath(out_path) && mkdir(out_path)
    # # action
    me==0 && println("Starting calculations 🚀")
    err_evo = []; iter_evo = []; t_tic = 0.0; ittot = 0
    ts = Float64[]; tt = 1.0; h5_names = String[]; isave = 1
    GC.enable(false) # uncomment for prof, mtp
    for it = 1:nt # time loop
        # CUDA.@profile for iter = 1:maxiter # uncomment for prof
        for iter = 1:maxiter # pseudo-transient loop
            if (it==1 && iter==11) t_tic = Base.time() end
            @parallel equal3!(VR_err, Vθ_err, Vφ_err, VR, Vθ, Vφ)
            @parallel compute_P!(P, DT_R, τ_RR, τ_θθ, τ_φφ, τ_Rθ, τ_Rφ, τ_θφ, VR, Vθ, Vφ, R, θ, η, ηSM, _dr, _dθ, _dφ, dmp, vpdt, lr)
            @parallel compute_TII!(τII, τ_RR, τ_θθ, τ_φφ, τ_Rθ, τ_Rφ, τ_θφ)
            @parallel (1:size(τII,2), 1:size(τII,3)) copy_bc_r!(τII)
            @parallel (1:size(τII,1), 1:size(τII,3)) copy_bc_θ!(τII)
            @parallel (1:size(τII,1), 1:size(τII,2)) copy_bc_φ!(τII)
            @parallel power_law!(η, τII, η_ini, s_ref, n_exp)
            @hide_communication b_width begin # comment for mtp
                @parallel compute_V_ηSM!(VR, Vθ, Vφ, DT_R, ηSM, η, τ_RR, τ_θθ, τ_φφ, τ_Rθ, τ_Rφ, τ_θφ, P, R, θ, ρ, _dr, _dθ, _dφ, g)
                @parallel (1:size(ηSM,2), 1:size(ηSM,3)) copy_bc_r!(ηSM)
                @parallel (1:size(ηSM,1), 1:size(ηSM,3)) copy_bc_θ!(ηSM)
                @parallel (1:size(ηSM,1), 1:size(ηSM,2)) copy_bc_φ!(ηSM)
                @parallel (1:size(VR,2), 1:size(VR,3))     bc_mx!(VR)
                @parallel (1:size(Vθ,2), 1:size(Vθ,3))     copy_bc_r!(Vθ)
                @parallel (1:size(Vφ,2), 1:size(Vφ,3))     copy_bc_r!(Vφ)
                @parallel (1:size(VR,1), 1:size(VR,3))     bc_0y!(VR)
                @parallel (1:size(Vθ,1), 1:size(Vθ,3))     bc_my!(Vθ)
                @parallel (1:size(Vφ,1), 1:(size(Vφ,3)-1)) bc_1!(Vφ, θ, r, sh)
                @parallel (1:size(Vφ,1))                   bc_2!(Vφ, θ, r, sh)
                @parallel (1:size(VR,1), 1:size(VR,2))     bc_0z!(VR)
                @parallel (1:size(Vθ,1), 1:size(Vθ,2))     bc_0z!(Vθ)
                @parallel (1:size(Vφ,1), 1:size(Vφ,2))     bc_3!(Vφ, θ, r, sh)
                update_halo!(VR, Vθ, Vφ, η, τII, ηSM) # comment for mtp
            end
            # if iter % nout == 0 # pseudo-transient loop exit criteria
            #     @parallel check_err!(VR_err, Vθ_err, Vφ_err, VR, Vθ, Vφ)
            #     err_vr = max_g(abs.(VR_err))./max_g(abs.(VR))
            #     err_vθ = max_g(abs.(Vθ_err))./max_g(abs.(Vθ))
            #     err_vφ = max_g(abs.(Vφ_err))./max_g(abs.(Vφ))
            #     err    = max(err_vr, err_vθ, err_vφ)
            #     # post-processing
            #     push!(err_evo, err); push!(iter_evo,iter)
            #     me==0 && @printf("iter %d, err=%1.3e \n", iter, err)
            #     if (err<εnonl && iter>20)  iter_end = iter; break; end # pseudo-transient loop exit criteria
            #     any(isnan.([err_vr,err_vθ,err_vφ])) && error("NaN")
            # end
            ittot = iter
        end
    end
    GC.enable(true) # uncomment for prof, mtp
    # Performance
    wtime    = Base.time() - t_tic
    A_eff    = (4*2 + 6*2 + 2 + 2*2)/1e9*nr*nθ*nφ*sizeof(Data.Number) # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(ittot-10)                                       # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                                         # Effective memory throughput [GB/s]
    me==0 && @printf("Total iters = %d (%d steps), time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", ittot, nt, wtime, round(T_eff, sigdigits=3))
    if do_save_p
        !ispath("../out_perf") && mkdir("../out_perf")
        open("../out_perf/out_SphericalStokes_pareff.txt","a") do io
            println(io, "$(nprocs) $(nr) $(nθ) $(nφ) $(ittot) $(wtime) $(A_eff) $(wtime_it) $(T_eff)")
        end
    end
    if do_save
        dim_g = (nr_g-2, nθ_g-2, nφ_g-2)
        out_h5 = joinpath(out_path,out_name)*"_$isave.h5"
        I = CartesianIndices(( (coords[1]*(nr-2) + 1):(coords[1]+1)*(nr-2),
                               (coords[2]*(nθ-2) + 1):(coords[2]+1)*(nθ-2),
                               (coords[3]*(nφ-2) + 1):(coords[3]+1)*(nφ-2) ))
        @parallel sph2cart!(X, Y, Z, R, θ, φ)
        fields = Dict("X"=>inn(X),"Y"=>inn(Y),"Z"=>inn(Z),
                      "eta"=>inn(η),"P"=>inn(P),"tII"=>inn(τII),"VR"=>av_x2i(VR),"VT"=>av_y2i(Vθ),"VF"=>av_z2i(Vφ),
                      "t_RR"=>inn(τ_RR),"t_TT"=>inn(τ_θθ),"t_FF"=>inn(τ_φφ),"t_RT"=>av_xya(τ_Rθ),"τ_RF"=>av_xza(τ_Rφ),"τ_TF"=>av_yza(τ_θφ))
        push!(ts,tt); push!(h5_names,out_name*"_$isave.h5")
        me==0 && print("Saving HDF5 file...")
        write_h5(out_h5,fields,dim_g,I,comm_cart,info) # comm_cart,MPI.Info() are varargs to exclude if using non-parallel HDF5 lib
        me==0 && println(" done")
        # write XDMF
        if me==0
            print("Saving XDMF file...")
            write_xdmf(joinpath(out_path,out_name)*".xdmf",h5_names,fields,dim_g,ts)
            println(" done")
        end
        isave += 1
    end
    finalize_global_grid()
    return
end

SphericalStokes()
