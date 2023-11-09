# DIAPIR - SPHERICAL COORDINATES SYSTEM
const USE_GPU   = haskey(ENV, "USE_GPU"  ) ? parse(Bool, ENV["USE_GPU"]  ) : true#false
const do_viz    = haskey(ENV, "DO_VIZ"   ) ? parse(Bool, ENV["DO_VIZ"]   ) : false
const do_save   = haskey(ENV, "DO_SAVE"  ) ? parse(Bool, ENV["DO_SAVE"]  ) : true
const do_save_p = haskey(ENV, "DO_SAVE_P") ? parse(Bool, ENV["DO_SAVE_P"]) : false
const nr        = haskey(ENV, "NR"       ) ? parse(Int , ENV["NR"]       ) : 208 - 1
const nθ        = haskey(ENV, "NTH"      ) ? parse(Int , ENV["NTH"]      ) : 208 - 1
const nφ        = haskey(ENV, "NPH"      ) ? parse(Int , ENV["NPH"]      ) : 208 - 1
const GPU_ID    = haskey(ENV, "GPU_ID"   ) ? parse(Int , ENV["GPU_ID"]   ) : 2
const dmp       = haskey(ENV, "DMP"      ) ? parse(Float64, ENV["DMP"]   ) : 4.5

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
function Save_infos(num, lr, lθ, lz, nr, nθ, nz, εnonl, runtime; out="../out_sph")
    fid=open(out * "/$(num)_infos.inf", "w")
    @printf(fid,"%d %f %f %f %d %d %d %d %d", num, lr, lθ, lz, nr, nθ, nz, εnonl, runtime); close(fid)
end

function Save_phys(num, η0, ρ0, ρ_in, n_exp, τ_C, vr, g, sh, r, radius, dmp; out="../out_sph")
    fid=open(out * "/$(num)_phys.inf", "w")
    @printf(fid,"%d   %f  %f  %f    %f     %f   %f  %f, %f, %f, %f    , %f" ,
                 num, η0, ρ0, ρ_in, n_exp, τ_C, vr, g , sh, r , radius, dmp); close(fid)
end

@static if USE_GPU
    function SaveArray(Aname, A; out="../out_sph")
        A_tmp = Array(A)
        fname = string(out, "/A_", Aname, ".bin");  fid = open(fname,"w"); write(fid, A_tmp); close(fid)
    end
else
    function SaveArray(Aname, A; out="../out_sph")
        fname = string(out, "/A_", Aname, ".bin"); fid = open(fname,"w"); write(fid, A); close(fid)
    end
end

# ===========================================================================================================
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

# BOUNDARY CONDITIONS =======================================================================================
@parallel_indices (ix,iy,iz) function BC_R!(VR::Data.Array, Vθ::Data.Array, Vφ::Data.Array)
    if (ix==1          && iy<=size(VR,2) && iz<=size(VR,3)) VR[ix,iy,iz] = -VR[ix+1,iy,iz]  end
    if (ix==size(VR,1) && iy<=size(VR,2) && iz<=size(VR,3)) VR[ix,iy,iz] = -VR[ix-1,iy,iz]  end
    if (ix==1          && iy<=size(Vθ,2) && iz<=size(Vθ,3)) Vθ[ix,iy,iz] =  Vθ[ix+1,iy,iz]  end
    if (ix==size(Vθ,1) && iy<=size(Vθ,2) && iz<=size(Vθ,3)) Vθ[ix,iy,iz] =  Vθ[ix-1,iy,iz]  end
    if (ix==1          && iy<=size(Vφ,2) && iz<=size(Vφ,3)) Vφ[ix,iy,iz] =  Vφ[ix+1,iy,iz]  end
    if (ix==size(Vφ,1) && iy<=size(Vφ,2) && iz<=size(Vφ,3)) Vφ[ix,iy,iz] =  Vφ[ix-1,iy,iz]  end
    return
end

@parallel_indices (ix,iy,iz) function BC_θ!(VR::Data.Array, Vθ::Data.Array , Vφ::Data.Array,
                                             θ::Data.Array,  r::Data.Number, sh::Data.Number)
    if (ix<=size(VR,1) && iy==1          && iz<=size(VR,3)  )   VR[ix,iy,iz] =  0.0                 end
    if (ix<=size(VR,1) && iy==size(VR,2) && iz<=size(VR,3)  )   VR[ix,iy,iz] =  0.0                 end
    if (ix<=size(Vθ,1) && iy==1          && iz<=size(Vθ,3)  )   Vθ[ix,iy,iz] = -Vθ[ix,iy+1,iz]      end
    if (ix<=size(Vθ,1) && iy==size(Vθ,2) && iz<=size(Vθ,3)  )   Vθ[ix,iy,iz] = -Vθ[ix,iy-1,iz]      end
    if (ix<=size(Vφ,1) && iy==1          && iz<=size(Vφ,3)-1)   Vφ[ix,iy,iz] = ( θ[ix,iy,iz  ] -
                                                                                 pi/2.0)*r*sh       end
    if (ix<=size(Vφ,1) && iy==1          && iz==size(Vφ,3)  )   Vφ[ix,iy,iz] = ( θ[ix,iy,iz-1] -
                                                                                 pi/2.0)*r*sh       end
    if (ix<=size(Vφ,1) && iy==size(Vφ,2) && iz<=size(Vφ,3)-1)   Vφ[ix,iy,iz] = ( θ[ix,iy,iz  ] -
                                                                                 pi/2.0)*r*sh       end
    if (ix<=size(Vφ,1) && iy==size(Vφ,2) && iz==size(Vφ,3)  )   Vφ[ix,iy,iz] = ( θ[ix,iy,iz-1] -
                                                                                 pi/2.0)*r*sh       end
    return
end

@parallel_indices (ix,iy,iz) function BC_φ!(VR::Data.Array, Vθ::Data.Array , Vφ::Data.Array,
                                             θ::Data.Array,  r::Data.Number, sh::Data.Number)
    if (ix<=size(VR,1) && iy<=size(VR,2) && iz==1         ) VR[ix,iy,iz] = 0.0                             end
    if (ix<=size(VR,1) && iy<=size(VR,2) && iz==size(VR,3)) VR[ix,iy,iz] = 0.0                             end
    if (ix<=size(Vθ,1) && iy<=size(Vθ,2) && iz==1         ) Vθ[ix,iy,iz] = 0.0                             end
    if (ix<=size(Vθ,1) && iy<=size(Vθ,2) && iz==size(Vθ,3)) Vθ[ix,iy,iz] = 0.0                             end
    if (ix<=size(Vφ,1) && iy<=size(Vφ,2) && iz==1         ) Vφ[ix,iy,iz] = 2*(θ[ix,iy,iz  ] -
                                                                           pi/2.0)*r*sh - Vφ[ix,iy,iz+1]   end
    if (ix<=size(Vφ,1) && iy<=size(Vφ,2) && iz==size(Vφ,3)) Vφ[ix,iy,iz] = 2*(θ[ix,iy,iz-1] -
                                                                           pi/2.0)*r*sh - Vφ[ix,iy,iz-1]   end
    return
end

# COPY BOUNDARIES ===========================================================================================
@parallel_indices (ix,iy,iz) function copy_bc_r!(A::Data.Array)
    if (ix==1         && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = A[ix+1,iy,iz] end
    if (ix==size(A,1) && iy<=size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = A[ix-1,iy,iz] end
    return
end

@parallel_indices (ix,iy,iz) function copy_bc_θ!(A::Data.Array)
    if (ix<=size(A,1) && iy==1         && iz<=size(A,3))    A[ix,iy,iz] = A[ix,iy+1,iz] end
    if (ix<=size(A,1) && iy==size(A,2) && iz<=size(A,3))    A[ix,iy,iz] = A[ix,iy-1,iz] end
    return
end

@parallel_indices (ix,iy,iz) function copy_bc_φ!(A::Data.Array)
    if (ix<=size(A,1) && iy<=size(A,2) && iz==1        )    A[ix,iy,iz] = A[ix,iy,iz+1] end
    if (ix<=size(A,1) && iy<=size(A,2) && iz==size(A,3))    A[ix,iy,iz] = A[ix,iy,iz-1] end
    return
end

function copy_BC!(A::Data.Array)
    @parallel copy_bc_r!(A)
    @parallel copy_bc_θ!(A)
    @parallel copy_bc_φ!(A)
end

# INITIALIZATION ============================================================================================
@parallel_indices (ix,iy,iz) function initialize_inclusion!( A::Data.Array ,      R::Data.Array ,
                                                             θ::Data.Array ,      φ::Data.Array,
                                                             r::Data.Number, radius::Data.Number,
                                                            in::Data.Number)
    if (ix<=size(A,1) && iy<=size(A,2) && iz<=size(A,3))
        if (((R[ix,iy,iz]-r)^2 + ((θ[ix,iy,iz]-pi/2.0)*r)^2 + (φ[ix,iy,iz]*r)^2) < radius)
            A[ix,iy,iz] = in
        end
    end
    return
end

@parallel_indices (ix,iy,iz) function initialize_velocity!(   V::Data.Array, COORD::Data.Array,
                                                           fact::Data.Number,    r::Data.Number)
    if (ix<=size(V,1) && iy<=size(V,2) && iz<=size(V,3)-1)  V[ix,iy,iz] = (COORD[ix,iy,iz  ] -
                                                                           pi/2.0)*r*fact   end
    if (ix<=size(V,1) && iy<=size(V,2) && iz==size(V,3)  )  V[ix,iy,iz] = (COORD[ix,iy,iz-1] -
                                                                           pi/2.0)*r*fact   end
    return
end

# ===========================================================================================================
@parallel function maxloc!(ηSM::Data.Array, η::Data.Array)
    @inn(ηSM) = @maxloc(η)
    return
end

# SOLVER ====================================================================================================
macro KBDT(ix,iy,iz)        esc(:(  dmp * 2.0 * pi * vpdt / lr * ηSM[$ix,$iy,$iz]   ))                    end
macro GSDT(ix,iy,iz)        esc(:(        4.0 * pi * vpdt / lr * ηSM[$ix,$iy,$iz]   ))                    end
macro avxa_VR(ix,iy,iz)     esc(:(  ((   VR[$ix  ,$iy  ,$iz  ] +    VR[$ix+1,$iy  ,$iz  ])*0.5) ))        end
macro avxi_Vθ(ix,iy,iz)     esc(:(  ((   Vθ[$ix  ,$iy+1,$iz+1] +    Vθ[$ix+1,$iy+1,$iz+1])*0.5) ))        end
macro avya_Vθ(ix,iy,iz)     esc(:(  ((   Vθ[$ix  ,$iy  ,$iz  ] +    Vθ[$ix  ,$iy+1,$iz  ])*0.5) ))        end
macro avxi_Vφ(ix,iy,iz)     esc(:(  ((   Vφ[$ix  ,$iy+1,$iz+1] +    Vφ[$ix+1,$iy+1,$iz+1])*0.5) ))        end
macro avyi_Vφ(ix,iy,iz)     esc(:(  ((   Vφ[$ix+1,$iy  ,$iz+1] +    Vφ[$ix+1,$iy+1,$iz+1])*0.5) ))        end
macro avxyi_R(ix,iy,iz)     esc(:(  ((    R[$ix  ,$iy  ,$iz+1] +     R[$ix  ,$iy+1,$iz+1] +
                                          R[$ix+1,$iy  ,$iz+1] +     R[$ix+1,$iy+1,$iz+1])*0.25)  ))      end
macro avxzi_R(ix,iy,iz)     esc(:(  ((    R[$ix  ,$iy+1,$iz  ] +     R[$ix  ,$iy+1,$iz+1] +
                                          R[$ix+1,$iy+1,$iz  ] +     R[$ix+1,$iy+1,$iz+1])*0.25)  ))      end
macro avyzi_R(ix,iy,iz)     esc(:(  ((    R[$ix+1,$iy  ,$iz  ] +     R[$ix+1,$iy  ,$iz+1] +
                                          R[$ix+1,$iy+1,$iz  ] +     R[$ix+1,$iy+1,$iz+1])*0.25)  ))      end
macro avxzi_θ(ix,iy,iz)     esc(:(  ((    θ[$ix  ,$iy+1,$iz  ] +     θ[$ix  ,$iy+1,$iz+1] +
                                          θ[$ix+1,$iy+1,$iz  ] +     θ[$ix+1,$iy+1,$iz+1])*0.25)  ))      end
macro avyzi_θ(ix,iy,iz)     esc(:(  ((    θ[$ix+1,$iy  ,$iz  ] +     θ[$ix+1,$iy  ,$iz+1] +
                                          θ[$ix+1,$iy+1,$iz  ] +     θ[$ix+1,$iy+1,$iz+1])*0.25)  ))      end
macro avxyi_GSDT(ix,iy,iz)  esc(:(  ((@GSDT($ix  ,$iy  ,$iz+1) + @GSDT($ix  ,$iy+1,$iz+1) +
                                      @GSDT($ix+1,$iy  ,$iz+1) + @GSDT($ix+1,$iy+1,$iz+1))*0.25)  ))      end
macro avxzi_GSDT(ix,iy,iz)  esc(:(  ((@GSDT($ix  ,$iy+1,$iz  ) + @GSDT($ix  ,$iy+1,$iz+1) +
                                      @GSDT($ix+1,$iy+1,$iz  ) + @GSDT($ix+1,$iy+1,$iz+1))*0.25)  ))      end
macro avyzi_GSDT(ix,iy,iz)  esc(:(  ((@GSDT($ix+1,$iy  ,$iz  ) + @GSDT($ix+1,$iy  ,$iz+1) +
                                      @GSDT($ix+1,$iy+1,$iz  ) + @GSDT($ix+1,$iy+1,$iz+1))*0.25)  ))      end
macro avxyi_η(ix,iy,iz)     esc(:(  ((    η[$ix  ,$iy  ,$iz+1] +     η[$ix  ,$iy+1,$iz+1] +
                                          η[$ix+1,$iy  ,$iz+1] +     η[$ix+1,$iy+1,$iz+1])*0.25)  ))      end
macro avxzi_η(ix,iy,iz)     esc(:(  ((    η[$ix  ,$iy+1,$iz  ] +     η[$ix  ,$iy+1,$iz+1] +
                                          η[$ix+1,$iy+1,$iz  ] +     η[$ix+1,$iy+1,$iz+1])*0.25)  ))      end
macro avyzi_η(ix,iy,iz)     esc(:(  ((    η[$ix+1,$iy  ,$iz  ] +     η[$ix+1,$iy  ,$iz+1] +
                                          η[$ix+1,$iy+1,$iz  ] +     η[$ix+1,$iy+1,$iz+1])*0.25)  ))      end
macro DIVV(ix,iy,iz)        esc(:((                     VR[$ix+1,$iy,$iz] - VR[$ix,$iy,$iz])*_dr +
                                    1.0/R[$ix,$iy,$iz]*(Vθ[$ix,$iy+1,$iz] - Vθ[$ix,$iy,$iz])*_dθ +
                                    1.0/R[$ix,$iy,$iz]/sin(θ[$ix,$iy,$iz])*
                                                       (Vφ[$ix,$iy,$iz+1] - Vφ[$ix,$iy,$iz])*_dφ +
                                    2.0*@avxa_VR(ix,iy,iz)/R[$ix,$iy,$iz] +
                                    @avya_Vθ(ix,iy,iz)/R[$ix,$iy,$iz]*cot(θ[$ix,$iy,$iz])   ))             end
macro DRR(ix,iy,iz)         esc(:(  (VR[$ix+1,$iy,$iz] - VR[$ix,$iy,$iz])*_dr - 1.0/3.0*@DIVV(ix,iy,iz) )) end
macro Dθθ(ix,iy,iz)         esc(:(  1.0/R[$ix,$iy,$iz]*(Vθ[$ix,$iy+1,$iz] - Vθ[$ix,$iy,$iz])*_dθ +
                                    @avxa_VR(ix,iy,iz)/R[$ix,$iy,$iz] - 1.0/3.0*@DIVV(ix,iy,iz) ))         end
macro Dφφ(ix,iy,iz)         esc(:(  1.0/R[$ix,$iy,$iz]/sin(θ[$ix,$iy,$iz])*
                                    (Vφ[$ix,$iy,$iz+1] - Vφ[$ix,$iy,$iz])*_dφ +
                                    @avxa_VR(ix,iy,iz)/R[$ix,$iy,$iz] +
                                    @avya_Vθ($ix,$iy,$iz)/R[$ix,$iy,$iz]*cot(θ[$ix,$iy,$iz]) -
                                    1.0/3.0*@DIVV(ix,iy,iz)  ))                                            end
macro DRθ(ix,iy,iz)         esc(:(                         (Vθ[$ix+1,$iy+1,$iz+1] - Vθ[$ix,$iy+1,$iz+1])*_dr +
                                    1.0/@avxyi_R(ix,iy,iz)*(VR[$ix+1,$iy+1,$iz+1] - VR[$ix+1,$iy,$iz+1])*_dθ -
                                    @avxi_Vθ(ix,iy,iz)/@avxyi_R(ix,iy,iz)   ))                             end
macro DRφ(ix,iy,iz)         esc(:(  (Vφ[$ix+1,$iy+1,$iz+1] - Vφ[$ix,$iy+1,$iz+1])*_dr +
                                    1.0/@avxzi_R(ix,iy,iz)/sin(@avxzi_θ(ix,iy,iz))*
                                    (VR[$ix+1,$iy+1,$iz+1] - VR[$ix+1,$iy+1,$iz])*_dφ -
                                    @avxi_Vφ(ix,iy,iz)/@avxzi_R(ix,iy,iz)   ))                             end
macro Dθφ(ix,iy,iz)         esc(:(  1.0/@avyzi_R(ix,iy,iz)*(Vφ[$ix+1,$iy+1,$iz+1] - Vφ[$ix+1,$iy,$iz+1])*_dθ+
                                    1.0/@avyzi_R(ix,iy,iz)/sin(@avyzi_θ(ix,iy,iz))*
                                    (Vθ[$ix+1,$iy+1,$iz+1] - Vθ[$ix+1,$iy+1,$iz])*_dφ -
                                    @avyi_Vφ(ix,iy,iz)/@avyzi_R(ix,iy,iz)*cot(@avyzi_θ(ix,iy,iz))  ))      end
@parallel_indices (ix,iy,iz) function compute_P!(   P::Data.Array , DT_R::Data.Array ,
                                                 τ_RR::Data.Array , τ_θθ::Data.Array , τ_φφ::Data.Array ,
                                                 τ_Rθ::Data.Array , τ_Rφ::Data.Array , τ_θφ::Data.Array ,
                                                   VR::Data.Array ,   Vθ::Data.Array ,   Vφ::Data.Array ,
                                                    R::Data.Array ,    θ::Data.Array ,    η::Data.Array ,
                                                  ηSM::Data.Array,
                                                  _dr::Data.Number,  _dθ::Data.Number,  _dφ::Data.Number,
                                                  dmp::Data.Number, vpdt::Data.Number,   lr::Data.Number)
    if (ix<=size(P,1) && iy<=size(P,2) && iz<=size(P,3))
        P[ix,iy,iz] = P[ix,iy,iz] - @KBDT(ix,iy,iz) * @DIVV(ix,iy,iz)
        τ_RR[ix,iy,iz] = (τ_RR[ix,iy,iz] + @GSDT(ix,iy,iz)*2.0*@DRR(ix,iy,iz))/
                                    (1.0 + @GSDT(ix,iy,iz)/η[ix,iy,iz])
        τ_θθ[ix,iy,iz] = (τ_θθ[ix,iy,iz] + @GSDT(ix,iy,iz)*2.0*@Dθθ(ix,iy,iz))/
                                    (1.0 + @GSDT(ix,iy,iz)/η[ix,iy,iz])
        τ_φφ[ix,iy,iz] = (τ_φφ[ix,iy,iz] + @GSDT(ix,iy,iz)*2.0*@Dφφ(ix,iy,iz))/
                                    (1.0 + @GSDT(ix,iy,iz)/η[ix,iy,iz])
        DT_R[ix,iy,iz] = vpdt^2 / (@KBDT(ix,iy,iz) + @GSDT(ix,iy,iz)/
                                    (1.0 + @GSDT(ix,iy,iz)/η[ix,iy,iz]))
    end
    if (ix<=size(τ_Rθ,1) && iy<=size(τ_Rθ,2) && iz<=size(τ_Rθ,3))
        τ_Rθ[ix,iy,iz] = (τ_Rθ[ix,iy,iz] + @avxyi_GSDT(ix,iy,iz)*@DRθ(ix,iy,iz))/
                                    (1.0 + @avxyi_GSDT(ix,iy,iz)/@avxyi_η(ix,iy,iz))
    end
    if (ix<=size(τ_Rφ,1) && iy<=size(τ_Rφ,2) && iz<=size(τ_Rφ,3))
        τ_Rφ[ix,iy,iz] = (τ_Rφ[ix,iy,iz] + @avxzi_GSDT(ix,iy,iz)*@DRφ(ix,iy,iz))/
                                    (1.0 + @avxzi_GSDT(ix,iy,iz)/@avxzi_η(ix,iy,iz))
    end
    if (ix<=size(τ_θφ,1) && iy<=size(τ_θφ,2) && iz<=size(τ_θφ,3))
        τ_θφ[ix,iy,iz] = (τ_θφ[ix,iy,iz] + @avyzi_GSDT(ix,iy,iz)*@Dθφ(ix,iy,iz))/
                                    (1.0 + @avyzi_GSDT(ix,iy,iz)/@avyzi_η(ix,iy,iz))
    end
    return
end

@parallel_indices (ix,iy,iz) function compute_TII!( τII::Data.Array,
                                                   τ_RR::Data.Array, τ_θθ::Data.Array, τ_φφ::Data.Array,
                                                   τ_Rθ::Data.Array, τ_Rφ::Data.Array, τ_θφ::Data.Array)
    if (ix<=size(τII,1)-2 && iy<=size(τII,2)-2 && iz<=size(τII,3)-2)
        τII[ix+1,iy+1,iz+1] = sqrt(1.0/2.0 * (τ_RR[ix+1,iy+1,iz+1]^2.0 + τ_θθ[ix+1,iy+1,iz+1]^2.0 +
                                τ_φφ[ix+1,iy+1,iz+1]^2.0) +
                              ((τ_Rθ[ix  ,iy  ,iz  ] + τ_Rθ[ix  ,iy+1,iz  ] +
                                τ_Rθ[ix+1,iy  ,iz  ] + τ_Rθ[ix+1,iy+1,iz  ])*0.25)^2.0 +
                              ((τ_Rφ[ix  ,iy  ,iz  ] + τ_Rφ[ix  ,iy  ,iz+1] +
                                τ_Rφ[ix+1,iy  ,iz  ] + τ_Rφ[ix+1,iy  ,iz+1])*0.25)^2.0 +
                              ((τ_θφ[ix  ,iy  ,iz  ] + τ_θφ[ix  ,iy  ,iz+1] +
                                τ_θφ[ix  ,iy+1,iz  ] + τ_θφ[ix  ,iy+1,iz+1])*0.25)^2.0)
    end
    return
end

@parallel_indices (ix,iy,iz) function power_law!(  η::Data.Array ,   τII::Data.Array , η_ini::Data.Array,
                                                 τ_C::Data.Number, n_exp::Data.Number)
    if (ix<=size(η,1) && iy<=size(η,2) && iz<=size(η,3))
        η[ix,iy,iz] = 2.0/(1.0/η_ini[ix,iy,iz] + 1.0/(η_ini[ix,iy,iz] * (τII[ix,iy,iz]/τ_C)^(1.0-n_exp)))
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
macro dVR(ix,iy,iz)         esc(:(                        (@σRR(ix+1,iy+1,iz+1) - @σRR(ix,iy+1,iz+1))*_dr +
                                    1.0/@avxi_R(ix,iy,iz)*(τ_Rθ[ix  ,iy+1,iz  ] - τ_Rθ[ix,iy  ,iz  ])*_dθ +
                                    1.0/@avxi_R(ix,iy,iz)/sin(@avxi_θ(ix,iy,iz))*
                                                          (τ_Rφ[ix  ,iy  ,iz+1] - τ_Rφ[ix,iy  ,iz  ])*_dφ +
                                    2.0*@avxi_σRR(ix,iy,iz)                     /@avxi_R(ix,iy,iz) -
                                       (@avxi_σθθ(ix,iy,iz)+@avxi_σφφ(ix,iy,iz))/@avxi_R(ix,iy,iz) +
                                        @avya_τRθ(ix,iy,iz)/@avxi_R(ix,iy,iz)*cot(@avxi_θ(ix,iy,iz)) -
                                        @avxi_ρG(ix,iy,iz)  ))  end
macro dVθ(ix,iy,iz)         esc(:(                        (   τ_Rθ[ix+1,iy  ,iz  ] - τ_Rθ[ix  ,iy,iz  ])*_dr +
                                    1.0/@avyi_R(ix,iy,iz)*(   @σθθ(ix+1,iy+1,iz+1) - @σθθ(ix+1,iy,iz+1))*_dθ +
                                    1.0/@avyi_R(ix,iy,iz)/sin(@avyi_θ(ix,iy,iz))*
                                                         (   τ_θφ[ix  ,iy  ,iz+1] - τ_θφ[ix  ,iy,iz  ])*_dφ +
                                    3.0*@avxa_τRθ(ix,iy,iz)                       /@avyi_R(ix  ,iy  ,iz  ) +
                                       (@avyi_σθθ(ix,iy,iz) - @avyi_σφφ(ix,iy,iz))/
                                        @avyi_R(ix  ,iy  ,iz  )*cot(@avyi_θ(ix,iy,iz)) ))  end
macro dVφ(ix,iy,iz)         esc(:(                        (τ_Rφ[ix+1,iy  ,iz  ] - τ_Rφ[ix  ,iy  ,iz])*_dr +
                                    1.0/@avzi_R(ix,iy,iz)*(τ_θφ[ix  ,iy+1,iz  ] - τ_θφ[ix  ,iy  ,iz])*_dθ +
                                    1.0/@avzi_R(ix,iy,iz)/sin(@avzi_θ(ix,iy,iz))*
                                                          (@σφφ(ix+1,iy+1,iz+1) - @σφφ(ix+1,iy+1,iz))*_dφ +
                                    3.0*@avxa_τRφ(ix,iy,iz)/@avzi_R(ix,iy,iz) +
                                    2.0*@avya_τθφ(ix,iy,iz)/@avzi_R(ix,iy,iz)*cot(@avzi_θ(ix,iy,iz))    )) end
@parallel_indices (ix,iy,iz) function compute_V!(  VR::Data.Array ,   Vθ::Data.Array ,   Vφ::Data.Array ,
                                                 DT_R::Data.Array ,
                                                 τ_RR::Data.Array , τ_θθ::Data.Array , τ_φφ::Data.Array ,
                                                 τ_Rθ::Data.Array , τ_Rφ::Data.Array , τ_θφ::Data.Array ,
                                                    P::Data.Array ,    R::Data.Array ,    θ::Data.Array ,
                                                    ρ::Data.Array ,
                                                  _dr::Data.Number,  _dθ::Data.Number,  _dφ::Data.Number,
                                                    g::Data.Number)
    if (ix<=size(VR,1)-2 && iy<=size(VR,2)-2 && iz<=size(VR,3)-2)
        VR[ix+1,iy+1,iz+1] =  VR[ix+1,iy+1,iz+1] + @dVR(ix,iy,iz)*
                            ((DT_R[ix,iy+1,iz+1] + DT_R[ix+1,iy+1,iz+1])*0.5)
    end
    if (ix<=size(Vθ,1)-2 && iy<=size(Vθ,2)-2 && iz<=size(Vθ,3)-2)
        Vθ[ix+1,iy+1,iz+1] =  Vθ[ix+1,iy+1,iz+1] + @dVθ(ix,iy,iz)*
                            ((DT_R[ix+1,iy,iz+1] + DT_R[ix+1,iy+1,iz+1])*0.5)
    end
    if (ix<=size(Vφ,1)-2 && iy<=size(Vφ,2)-2 && iz<=size(Vφ,3)-2)
        Vφ[ix+1,iy+1,iz+1] =  Vφ[ix+1,iy+1,iz+1] + @dVφ(ix,iy,iz)*
                            ((DT_R[ix+1,iy+1,iz] + DT_R[ix+1,iy+1,iz+1])*0.5)
    end
    return
end

# CHECK ERROR ===============================================================================================
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

# ===========================================================================================================
@views function SPH_3D_Inclusion()
    num      = 1
    # physical parameters -----------------------------------------------------------------------------------
    η0       = 1.0              # Pa*s  , media viscosity
    vr       = 1e2              #         viscosity ratio out/in
    ρ0       = 0.0              # kg/m^3, media density
    ρ_in     = ρ0 - 10.0        # kg/m^3, inclusion density
    g        = 1.0              # m/s^2 , gravity acceleration
    n_exp    = 5.0              #         power law exponent
    τ_C      = 1.0              # Pa    , characteristic stress
    r        = 1000.0           # m     , radius of the total sphere
    lr       = 6.0              # m     , model dimension in r
    lθ, lφ   = lr/r, lr/r       # m     , model dimension in θ and φ
    radius   = 1.0              # m     , radius of the inclusion
    sh       = 1.0              # m/s   , shearing velocity
    # numerics ----------------------------------------------------------------------------------------------
    εnonl    = 5e-7                         # pseudo-transient loop exit criteria
    nt       = 1                            # number of time steps
    maxiter  = 1e5                          # maximum number of pseudo-transient iterations
    nout     = 1e2                          # pseudo-transient plotting frequency
    CFL      = 1.0/(2.0 + 4.5*log10(vr))    # Courant-Friedrichs-Lewy condition
    # preprocessing -----------------------------------------------------------------------------------------
     dr           = lr/(nr-1)
    _dr, _dθ, _dφ = (nr-1)/lr, (nθ-1)/lθ, (nφ-1)/lφ
    R_i      = range(     r - lr/2.0,      r + lr/2.0, length=nr)    # coordinates of grid points in r
    θ_i      = range(pi/2.0 - lθ/2.0, pi/2.0 + lθ/2.0, length=nθ)    # coordinates of grid points in θ
    φ_i      = range(       - lφ/2.0,          lφ/2.0, length=nφ)    # coordinates of grid points in z
    (R,θ,φ)  = ([xc for xc = R_i, yc = θ_i, zc = φ_i],[yc for xc = R_i, yc = θ_i, zc = φ_i],
                [zc for xc = R_i, yc = θ_i, zc = φ_i])               # grid of coordinates
    R        = Data.Array(R)
    θ        = Data.Array(θ)
    φ        = Data.Array(φ)
    # initialization and boundary conditions ----------------------------------------------------------------
    print("Starting initialization ... ")
    P        =   @zeros(nr  , nθ  , nφ  )
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
    vpdt     = dr*CFL
    @parallel initialize_inclusion!(ρ, R, θ, φ, r, radius, ρ_in)
    @parallel initialize_inclusion!(η, R, θ, φ, r, radius, η0/vr)
    @parallel initialize_velocity!(Vφ, θ, sh, r)
    η_ini   .= η
    println("done.")
    # action ------------------------------------------------------------------------------------------------
    println("Starting calculation (nr=$nr, nθ=$nθ , nφ=$nφ)")
    err_evo = []; iter_evo = []; t_tic = 0.0; ittot = 0
    for it = 1:nt # time loop
        for iter = 1:maxiter # pseudo-transient loop
            if (it==1 && iter==11) GC.gc(); t_tic = Base.time() end
            # SOLVER ----------------------------------------------------------------------------------------
            @parallel equal3!(VR_err, Vθ_err, Vφ_err, VR, Vθ, Vφ)
            @parallel maxloc!(ηSM, η)
            copy_BC!(ηSM)
            @parallel compute_P!(P, DT_R, τ_RR, τ_θθ, τ_φφ, τ_Rθ, τ_Rφ, τ_θφ, VR, Vθ, Vφ, R, θ,
                                 η, ηSM , _dr , _dθ , _dφ , dmp , vpdt, lr)
            @parallel compute_TII!(τII, τ_RR, τ_θθ, τ_φφ, τ_Rθ, τ_Rφ, τ_θφ)
            copy_BC!(τII)
            @parallel power_law!(η, τII, η_ini, τ_C, n_exp)
            @parallel compute_V!(VR , Vθ , Vφ , DT_R, τ_RR, τ_θθ, τ_φφ, τ_Rθ, τ_Rφ, τ_θφ, P, R, θ, ρ,
                                 _dr, _dθ, _dφ, g)
            @parallel BC_R!(VR, Vθ, Vφ)
            @parallel BC_θ!(VR, Vθ, Vφ, θ, r, sh)
            @parallel BC_φ!(VR, Vθ, Vφ, θ, r, sh)
            # pseudo-transient loop exit criteria -----------------------------------------------------------
            if iter % nout == 0
                @parallel check_err!(VR_err, Vθ_err, Vφ_err, VR, Vθ, Vφ)
                err_vr = maximum(abs.(VR_err))./maximum(abs.(VR))
                err_vθ = maximum(abs.(Vθ_err))./maximum(abs.(Vθ))
                err_vφ = maximum(abs.(Vφ_err))./maximum(abs.(Vφ))
                err    = max(err_vr, err_vθ, err_vφ)
                # post-processing
                push!(err_evo, err)
                push!(iter_evo,iter)
                if do_viz  display(scatter(iter_evo, err_evo, xaxis=:log,yaxis=:log,legend=false))  end
                @printf("iter %d, err=%1.3e \n", iter, err)
                # pseudo-transient loop exit criteria
                if (err<εnonl && iter>20)  iter_end = iter; break; end
            end
            ittot = iter
        end
    end
    # Performance -------------------------------------------------------------------------------------------
    wtime    = Base.time() - t_tic
    # Effective main memory access per iteration [GB]
    A_eff    = (4*2 + 6*2 + 2 + 2*2)/1e9*nr*nθ*nφ*sizeof(Data.Number)
    # (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess;
    # Ci has to be read : 1 whole-array memaccess)
    wtime_it = wtime/(ittot-10)                                       # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                                         # Effective memory throughput [GB/s]
    @printf("Total iters = %d (%d steps), time = %1.3e sec (@ T_eff = %1.2f GB/s) \n",
            ittot, nt, wtime, round(T_eff, sigdigits=3))
    if do_save_p
        !ispath("./out_perf") && mkdir("./out_perf")
        open("./out_perf/out_SPH_3D_PERF.txt","a") do io
            println(io, "$(nr) $(nθ) $(nφ) $(dmp) $(ittot) $(wtime) $(A_eff) $(wtime_it) $(T_eff)")
        end
    end
    # SAVING ------------------------------------------------------------------------------------------------
    if do_save
        !ispath("../out_sph") && mkdir("../out_sph")
        err_evo  = Data.Array(err_evo)
        iter_evo = Data.Array(iter_evo)
        Save_phys(num, η0, ρ0, ρ_in, n_exp, τ_C, vr, g, sh, r, radius, dmp)
        Save_infos(num, lr, lθ, lφ, nr, nθ, nφ, εnonl, wtime)
        SaveArray("err_evo" , err_evo )
        SaveArray("iter_evo", iter_evo)
        SaveArray("R"       , R       )
        SaveArray("TH"      , θ       )
        SaveArray("PH"      , φ       )
        SaveArray("ETAS"    , η       )
        SaveArray("RHO"     , ρ       )
        SaveArray("P"       , P       )
        SaveArray("VR"      , VR      )
        SaveArray("VTH"     , Vθ      )
        SaveArray("VPH"     , Vφ      )
        SaveArray("TAU_RR"  , τ_RR    )
        SaveArray("TAU_THTH", τ_θθ    )
        SaveArray("TAU_PHPH", τ_φφ    )
        SaveArray("TAU_RTH" , τ_Rθ    )
        SaveArray("TAU_RPH" , τ_Rφ    )
        SaveArray("TAU_THPH", τ_θφ    )
        SaveArray("TII"     , τII     )
    end
    return
end

@time SPH_3D_Inclusion()
