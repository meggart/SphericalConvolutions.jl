module SphericalConvolutions
using LinearAlgebra: lmul!, ldiv!
using FastTransforms

struct PLM{T}
  coefs::Matrix{T}
end
function lm2ij(l,m)
  l<0 && throw(BoundsError("l must be positive"))
  abs(m)>l && throw(BoundsError("abs(m) must be <= l"))
  row = l - (abs(m))+1
  col = 2*abs(m) - (m<0)+1
  row,col
end
function Base.getindex(a::PLM,l::Integer,m::Integer)
  row,col=lm2ij(l,m)
  a.coefs[row,col]
end
function Base.setindex!(a::PLM,v,l,m)
  row,col=lm2ij(l,m)
  a.coefs[row,col]=v
end
lmax(c::PLM)=size(c.coefs,1)-1
mmax(c::PLM,l)=min(size(c.coefs,2)÷2,l)
export lmax,mmax


struct PlanAll{T,A,S,P}
  PA::A
  PS::S
  SHTP::P
  four2d::Matrix{T}
end
function PlanAll(x::Matrix)
  PA = FastTransforms.plan_sph_analysis(x)
  PS = FastTransforms.plan_sph_synthesis(x)
  four2d=zeros(size(x))
  p = FastTransforms.plan_sph2fourier(four2d)
  PlanAll(PA,PS,p,four2d)
end
function do_transform!(coefs::PLM,x::Matrix,plan::PlanAll)
  #fill!(plan.four2d,0.0);fill!(coefs.coefs,0.0)
  #f2d = plan.PA * x
  #c = plan.SHTP \ f2d
  copy!(coefs.coefs,x)
  lmul!(plan.PA,coefs.coefs)
  ldiv!(plan.SHTP,coefs.coefs)
  #A_mul_B!(plan.four2d,plan.PA,x)
  #At_mul_B!(coefs.coefs,plan.SHTP,plan.four2d)
  coefs
end

function do_transform(x::Matrix)
  coefs=PLM(zeros(size(x)))
  plan=PlanAll(x)
  do_transform!(coefs,x,plan)
end

function do_back_transform!(xout::Matrix,coefs::PLM,plan::PlanAll)
  #fill!(xout,0.0);fill!(plan.four2d,0.0)
  #f2d = plan.SHTP * coefs.coefs
  #x = plan.PS * f2d
  #copy!(xout,x)
  copy!(xout,coefs.coefs)
  lmul!(plan.SHTP,xout)
  lmul!(plan.PS,xout)
  #A_mul_B!(plan.four2d,plan.SHTP,coefs.coefs)
  #A_mul_B!(xout,plan.PS,plan.four2d)
  xout
end
function do_back_transform(coefs::PLM)
  x=zeros(size(coefs.coefs))
  plan=PlanAll(x)
  do_back_transform!(x,coefs,plan)
end
export PlanAll, do_transform, do_back_transform, do_transform!, do_back_transform!

function imageExpSpectrum(;beta=20,nlat=720,nlon=1439)
  c=PLM(zeros(nlat,nlon))
  for l=1:(nlat-1),m=-l:l
    c[l,m]=randn()*exp(-l/(beta))/sqrt(2*l+1)
  end
  im = do_back_transform(c)
end
function imagePowerSpectrum(;beta=1.5,nlat=720,nlon=1439)
  c=PLM(zeros(nlat,nlon))
  for l=1:(nlat-1),m=-l:l
    c[l,m]=randn()/l^(beta)/sqrt(2*l+1)
  end
  im = do_back_transform(c)
end
export imageExpSpectrum, imagePowerSpectrum

struct WaveletPlan{T,P,ET}
  r::T
  c::Vector{PLM{ET}}
  plan::P
  cmapped::Vector{PLM{ET}}
  cbuf::PLM{ET}
end
function WaveletPlan(im;r=1e6:2e6:9e6,kernelfunc = x->exp(-x^2))
  plan,c = generatekernels(nlat=size(im,1),nlon=size(im,2),r=r,kernelfunc=kernelfunc)
  cmapped = [PLM(zeros(size(im))) for i=1:length(r)]
  cbuf = PLM(zeros(size(im)))
  WaveletPlan(r,c,plan,cmapped,cbuf)
end

i2lon(i,nlon)=(i-0.5)*(360/nlon)-180
i2lat(i,nlat)=90-(i-0.5)*(180/nlat)
poledist(lat) = (90-lat)*40000/360
function makespot(nlat,nlon;r=1e6,kernelfunc = x->exp(-x^2))
  o = [kernelfunc(poledist(i2lat(j,nlat))/r) for j=1:nlat,_=1:nlon]
  o/sum(abs2,o)
end
function generatekernels(;nlat=721,nlon=1441,r=2e3:2e3:10e3, kernelfunc = x->exp(-x^2))
  kernelims = map(ir->makespot(nlat,nlon,r=ir;kernelfunc),r)
  plan = PlanAll(zeros(nlat,nlon))
  coefs = [PLM(zeros(nlat,nlon)) for i=1:length(r)]
  map(kernelims,coefs) do k,c
    do_transform!(c,k,plan)
    c.coefs[:]=c.coefs/c[0,0]
    for l=1:lmax(c),m=1:mmax(c,l)
      c[l,m] = c[l,0]
      c[l,-m]= c[l,0]
    end
  end
  plan,coefs
end
function wavelet!(imback::Vector,im::Matrix,wp::WaveletPlan)
  #Transform input image
  fill!(wp.cbuf.coefs,0.0)
  do_transform!(wp.cbuf,im,wp.plan)
  foreach(imback,wp.cmapped,wp.c) do x,cm,ci
    #Multiply with kernel
    for i in eachindex(cm.coefs)
      cm.coefs[i] = ci.coefs[i].*wp.cbuf.coefs[i]
    end
    #Backtransform
    do_back_transform!(x,cm,wp.plan)
  end
  imback
end
function gausswavelet(im;r=1e6:2e6:9e6)
  wp = WaveletPlan(im,r=r)
  imback = [zeros(size(im)) for i in r]
  gausswavelet!(imback,im,wp)
end

lowpasskernel(;lbord = 10, width = lbord/20) = l-> tanh((lbord-l)/width)/2.0+0.5


function lowpass(im;lbord = 10, width = lbord/20, scalef = lowpasskernel(;lbord,width))
  clm = do_transform(im)
  nlat = size(im,1)
  for il=1:(nlat-1)
    sc = scalef(il)
    for m=-il:il
      clm[il,m]=clm[il,m]*sc
    end
  end
  do_back_transform(clm)
end
export WaveletPlan, gausswavelet, gausswavelet!, lowpass

function sphericalspectrum(im)
  coefs = do_transform(im)
  map(1:lmax(coefs)) do l
      mm = mmax(coefs,l)
      sum(m->abs2(coefs[l,m]),-mm:mm)
  end
end
export sphericalspectrum

end

