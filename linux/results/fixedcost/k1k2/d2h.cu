#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
using C = std::chrono::steady_clock;
static double ms(C::time_point a){return std::chrono::duration<double,std::milli>(C::now()-a).count();}
const size_t B = 28u<<20;
struct Dev { void* d; cudaStream_t s; std::vector<unsigned char> h; void* pin; };
Dev dv[4];
void pageable(int k){ cudaSetDevice(k); cudaMemcpyAsync(dv[k].h.data(), dv[k].d, B, cudaMemcpyDefault, dv[k].s); cudaStreamSynchronize(dv[k].s);}
void staged(int k){ cudaSetDevice(k); const size_t CH=4u<<20; unsigned char* p=(unsigned char*)dv[k].pin;
  // chunked double buffer: D2H chunk i+1 while memcpy chunk i
  cudaEvent_t ev[2]; cudaEventCreateWithFlags(&ev[0],cudaEventDisableTiming); cudaEventCreateWithFlags(&ev[1],cudaEventDisableTiming);
  size_t n=(B+CH-1)/CH;
  auto issue=[&](size_t i){ size_t o=i*CH, l=std::min(CH,B-o); cudaMemcpyAsync(p+(i%2)*CH,(char*)dv[k].d+o,l,cudaMemcpyDefault,dv[k].s); cudaEventRecord(ev[i%2],dv[k].s);};
  issue(0);
  for(size_t i=0;i<n;i++){ if(i+1<n) issue(i+1); cudaEventSynchronize(ev[i%2]); size_t o=i*CH,l=std::min(CH,B-o); memcpy(dv[k].h.data()+o,p+(i%2)*CH,l);} 
  cudaEventDestroy(ev[0]); cudaEventDestroy(ev[1]);}
int main(){
  for(int k=0;k<4;k++){cudaSetDevice(k); cudaMalloc(&dv[k].d,B); cudaMemset(dv[k].d,1,B); cudaStreamCreateWithFlags(&dv[k].s,cudaStreamNonBlocking); dv[k].h.assign(B,0); cudaMallocHost(&dv[k].pin,8u<<20);} 
  for(int rep=0;rep<3;rep++){
    auto t=C::now(); for(int k=0;k<4;k++) pageable(k); double a=ms(t);
    t=C::now(); {std::vector<std::thread> th; for(int k=1;k<4;k++) th.emplace_back(pageable,k); pageable(0); for(auto&x:th)x.join();} double b=ms(t);
    t=C::now(); for(int k=0;k<4;k++) staged(k); double c=ms(t);
    t=C::now(); {std::vector<std::thread> th; for(int k=1;k<4;k++) th.emplace_back(staged,k); staged(0); for(auto&x:th)x.join();} double d=ms(t);
    printf("pageable serial %.1f parallel %.1f | staged serial %.1f parallel %.1f ms (4 x 28 MiB)\n",a,b,c,d);
  }
}
