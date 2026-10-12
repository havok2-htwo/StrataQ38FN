#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>
using C = std::chrono::steady_clock;
static double ms(C::time_point a){return std::chrono::duration<double,std::milli>(C::now()-a).count();}
int main(int argc,char**argv){
  int A=atoi(argv[1]),B=atoi(argv[2]); size_t rows=atoi(argv[3]); size_t D=10240; size_t bytes=rows*D*4;
  int can=0; cudaDeviceCanAccessPeer(&can,B,A); printf("canAccessPeer %d->%d: %d  bytes %.1f MB\n",B,A,can,bytes/1e6);
  float *dA,*dB,*h; cudaSetDevice(A); cudaMalloc(&dA,bytes); cudaStream_t sA; cudaStreamCreateWithFlags(&sA,cudaStreamNonBlocking);
  cudaSetDevice(B); cudaMalloc(&dB,bytes); cudaStream_t sB; cudaStreamCreateWithFlags(&sB,cudaStreamNonBlocking);
  cudaHostAlloc(&h,bytes,cudaHostAllocPortable);
  std::vector<cudaEvent_t> ev(16); cudaSetDevice(A); for(auto&e:ev) cudaEventCreateWithFlags(&e,cudaEventDisableTiming);
  cudaEvent_t t0,t1; cudaSetDevice(B); cudaEventCreate(&t1); cudaSetDevice(A); cudaEventCreate(&t0);
  for(int mode=0;mode<6;++mode){ double best=1e9,sum=0; int R=30;
   for(int r=0;r<R;++r){ cudaSetDevice(A); cudaDeviceSynchronize(); cudaSetDevice(B); cudaDeviceSynchronize();
    auto st=C::now();
    if(mode==0){ cudaSetDevice(A); cudaMemcpyAsync(h,dA,bytes,cudaMemcpyDeviceToHost,sA); cudaStreamSynchronize(sA);
      cudaSetDevice(B); cudaMemcpyAsync(dB,h,bytes,cudaMemcpyHostToDevice,sB); cudaStreamSynchronize(sB);}
    else if(mode==5){ cudaMemcpyPeerAsync(dB,B,dA,A,bytes,sB); cudaSetDevice(B); cudaStreamSynchronize(sB);}
    else { int P = mode==1?2: mode==2?4: mode==3?8:16; size_t per=(rows+P-1)/P;
      for(int i=0;i<P;++i){ size_t r0=i*per, n=std::min(rows,r0+per)-r0; cudaSetDevice(A);
        cudaMemcpyAsync(h+r0*D,dA+r0*D,n*D*4,cudaMemcpyDeviceToHost,sA); cudaEventRecord(ev[i],sA); }
      cudaSetDevice(B);
      for(int i=0;i<P;++i){ size_t r0=i*per, n=std::min(rows,r0+per)-r0; cudaStreamWaitEvent(sB,ev[i],0);
        cudaMemcpyAsync(dB+r0*D,h+r0*D,n*D*4,cudaMemcpyHostToDevice,sB);}
      cudaStreamSynchronize(sB);}
    double t=ms(st); if(r>2){best=std::min(best,t); sum+=t;}
   }
   const char* nm[]={"serial D2H,sync,H2D","pieces 2","pieces 4","pieces 8","pieces 16","cudaMemcpyPeerAsync"};
   printf("%-22s best %.3f ms  mean %.3f ms  err=%s\n",nm[mode],best,sum/(R-3),cudaGetErrorString(cudaGetLastError()));
  }
}
