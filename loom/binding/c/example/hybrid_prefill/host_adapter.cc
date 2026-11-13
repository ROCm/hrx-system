#include "iree/hal/api.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/io/file_handle.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

#define HYBRID_RETURN_IF_ERROR(...) do { auto hybrid_status = (__VA_ARGS__); if (!iree_status_is_ok(hybrid_status)) { fprintf(stderr,"HYBRID_ERROR line=%d\n",__LINE__); iree_status_fprint(stderr,hybrid_status); fflush(stderr); return hybrid_status; } } while (0)

namespace {
template<class T> T next(const char* name) { return reinterpret_cast<T>(dlsym(RTLD_NEXT,name)); }
thread_local bool internal=false;
struct Guard { bool prior=internal; Guard(){internal=true;} ~Guard(){internal=prior;} };
using Replay=std::function<iree_status_t(iree_hal_command_buffer_t*)>;
struct Op {
  Replay replay;
  bool split=false;
  iree_hal_executable_t* executable=nullptr;
  iree_hal_executable_function_t function={};
  std::array<iree_hal_buffer_ref_t,4> refs={};
};
struct Recording {std::vector<Op> ops; bool split=false;};
std::mutex records_mutex,run_mutex;
std::map<iree_hal_command_buffer_t*,std::shared_ptr<Recording>> records;
iree_hal_device_t* gpu=nullptr;
iree_hal_device_t* npu=nullptr;
iree_hal_device_group_t* group=nullptr;
iree_hal_pool_t* pool=nullptr;
iree_hal_buffer_t* scratch=nullptr;
iree_hal_queue_t* gq=nullptr;
iree_hal_queue_t* nq=nullptr;
iree_hal_semaphore_t* gs=nullptr;
iree_hal_semaphore_t* ns=nullptr;
iree_hal_executable_t* ne=nullptr;
iree_hal_executable_function_t nf={};
uint64_t gv=0,nv=0;
constexpr size_t M=1024,K=5120,N=17408,BANK=512;
size_t NN=0,GN=0,XN=0,WN=0,YN=0,YG=0,TOTAL=0;
size_t split_count=0;
struct Weight { std::vector<int8_t> packed, residual; std::vector<float> scales, residual_scales; };
std::map<std::pair<iree_hal_buffer_t*,size_t>,Weight> weights;

iree_hal_profile_sink_t* profile_sinks[2]={};
int profile_state=0;
std::mutex profile_mutex;
iree_status_t profile_control() {
  const char* prefix=getenv("LOOM_HYBRID_PROFILE");if(!prefix||!gpu||!npu)return iree_ok_status();
  std::lock_guard<std::mutex> lock(profile_mutex);
  if(profile_state==0&&std::ifstream(std::string(prefix)+".start").good()) {
    iree_hal_device_t* devices[]={gpu,npu};
    for(size_t i=0;i<2;++i) {
      const std::string path=std::string(prefix)+(i?"-npu.irpf":"-gpu.irpf");
      iree_io_file_handle_t* file=nullptr;
      HYBRID_RETURN_IF_ERROR(iree_io_file_handle_create(IREE_IO_FILE_MODE_WRITE,iree_make_cstring_view(path.c_str()),0,iree_allocator_system(),&file));
      auto status=iree_hal_profile_file_sink_create(file,iree_allocator_system(),&profile_sinks[i]);iree_io_file_handle_release(file);
      HYBRID_RETURN_IF_ERROR(status);
      iree_hal_device_profiling_options_t options={};options.sink=profile_sinks[i];
      options.data_families=IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS|IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
      options.data_families|=i?(IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS|IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES):IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
      HYBRID_RETURN_IF_ERROR(iree_hal_device_profiling_begin(devices[i],&options));
    }
    profile_state=1;fprintf(stderr,"HYBRID_PROFILE_BEGIN\n");fflush(stderr);
  }
  if(profile_state==1&&std::ifstream(std::string(prefix)+".stop").good()) {
    HYBRID_RETURN_IF_ERROR(iree_hal_device_profiling_end(gpu));
    HYBRID_RETURN_IF_ERROR(iree_hal_device_profiling_end(npu));
    for(auto*& sink:profile_sinks){iree_hal_profile_sink_release(sink);sink=nullptr;}
    profile_state=2;fprintf(stderr,"HYBRID_PROFILE_END\n");fflush(stderr);
  }
  return iree_ok_status();
}

void append(iree_hal_command_buffer_t* cb,Op op) {
  if(internal) return;
  std::lock_guard<std::mutex> lock(records_mutex);
  auto it=records.find(cb);
  if(it!=records.end()) {it->second->split|=op.split;it->second->ops.push_back(std::move(op));}
}
iree_status_t wait(iree_hal_semaphore_t* s,uint64_t v) {
  return iree_hal_semaphore_wait(s,v,iree_make_timeout_ms(120000),0);
}
iree_hal_buffer_ref_t resolve(iree_hal_buffer_ref_t r,iree_hal_buffer_binding_table_t table) {
  if(!r.buffer && r.buffer_slot<table.count) {
    const auto& b=table.bindings[r.buffer_slot];r.buffer=b.buffer;r.offset+=b.offset;
  }
  return r;
}
size_t aindex(size_t row,size_t col) {
  return (row/128)*128*K+(col/128)*16384+(row%128/8)*1024+(col%128/8)*64+(row%8)*8+col%8;
}
size_t bindex(size_t row,size_t col) {
  return (row/128)*128*K+(col/128)*16384+(row%128/8)*1024+(col%128/8)*64+(col%8)*8+row%8;
}
size_t cindex(size_t row,size_t col) {
  return ((row/128)*(NN/128)+col/128)*16384+(row%128/8)*1024+(col%128/8)*64+(row%8)*8+col%8;
}
float half(const uint8_t* p) {_Float16 h;memcpy(&h,p,2);return float(h);}
float dequant(const uint8_t* b,size_t k) {
  const size_t q=k/32,lo=q%4;
  const int scale=q<4 ? b[4+q]&63 : (b[12+lo]&15)|((b[4+lo]>>2)&48);
  const int minimum=q<4 ? b[8+q]&63 : (b[12+lo]>>4)|((b[8+lo]>>2)&48);
  const int v=(b[16+(q/2)*32+k%32] >> ((q%2)*4))&15;
  const float ds=float(_Float16(half(b)*scale));
  const float dm=float(_Float16(-half(b+2)*minimum));
  return float(_Float16(std::fma(ds,float(v),dm)));
}
iree_status_t transfer(iree_hal_buffer_ref_t r,void* data,size_t length,bool upload) {
  iree_hal_transfer_operation_t op={};
  if(upload) {op.type=IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;op.update.source_buffer=data;
    op.update.target_buffer=r.buffer;op.update.target_offset=r.offset;op.update.length=length;}
  else {op.type=IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;op.download.source_buffer=r.buffer;
    op.download.source_offset=r.offset;op.download.target=data;op.download.length=length;}
  uint64_t before=gv,after=++gv;
  iree_hal_semaphore_t* deps[]={gs,ns};uint64_t vals[]={before,nv};
  HYBRID_RETURN_IF_ERROR(iree_hal_queue_transfer(gq,{nv?size_t(2):size_t(1),deps,vals},{1,&gs,&after},1,&op,nullptr));
  return wait(gs,after);
}
iree_status_t initialize() {
  if(pool) return iree_ok_status();
  if(!group||!npu||!gpu) fprintf(stderr,"HYBRID_MISSING group=%p gpu=%p npu=%p\n",(void*)group,(void*)gpu,(void*)npu);
  if(!group||!npu||!gpu) return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,"hybrid device group not initialized");
  const char* f=getenv("LOOM_HYBRID_FRACTION");const double fraction=f?strtod(f,nullptr):0.15;
  if(!std::isfinite(fraction)||fraction<=0||fraction>=1) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,"fraction must be finite and in (0,1)");
  NN=size_t(std::round(N*fraction/512))*512;GN=N-NN;
  if(!NN||NN>=N) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,"aligned offload leaves an empty branch");
  XN=0;WN=M*K;YN=WN+NN*K;YG=YN+M*NN*4;TOTAL=YG+M*GN*4;
  gq=iree_hal_device_queue(gpu,0,0);nq=iree_hal_device_queue(npu,0,0);
  HYBRID_RETURN_IF_ERROR(iree_hal_semaphore_create(gpu,IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,0,IREE_HAL_SEMAPHORE_FLAG_NONE,&gs));
  HYBRID_RETURN_IF_ERROR(iree_hal_semaphore_create(npu,IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,0,IREE_HAL_SEMAPHORE_FLAG_NONE,&ns));
  iree_hal_pool_family_access_t families[2]={};
  families[0].family=iree_hal_queue_family(gq);families[1].family=iree_hal_queue_family(nq);
  for(auto& f:families) f.usage=IREE_HAL_BUFFER_USAGE_STORAGE|IREE_HAL_BUFFER_USAGE_TRANSFER;
  families[0].interfaces=UINT64_C(1)<<IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  families[1].interfaces=UINT64_C(1)<<IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA;
  iree_hal_pool_scope_t scope={2,families,{}};
  iree_hal_slab_pool_options_t options;iree_hal_slab_pool_options_initialize(&options);
  HYBRID_RETURN_IF_ERROR(iree_hal_slab_pool_create(group,scope,&options,iree_allocator_system(),&pool));
  HYBRID_RETURN_IF_ERROR(iree_hal_pool_allocate_buffer(pool,iree_hal_make_pool_reservation_request(TOTAL,64),iree_make_timeout_ms(120000),&scratch));
  const char* path=getenv("LOOM_HYBRID_IMAGE");
  if(!path) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,"LOOM_HYBRID_IMAGE is required");
  std::ifstream file(path,std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(file)),{});
  if(bytes.empty()) return iree_make_status(IREE_STATUS_NOT_FOUND,"NPU image is empty or absent");
  iree_hal_executable_target_selection_t selection={};selection.family=IREE_SV("xdna");selection.kind_flags=IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT;
  auto selected=iree_hal_device_spec_select_executable_target(iree_hal_device_spec(npu),&selection);
  if(selected.outcome!=IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) return iree_make_status(IREE_STATUS_UNAVAILABLE,"NPU target missing");
  iree_hal_executable_load_params_t params;iree_hal_executable_load_params_initialize(&params);
  params.executable_data=iree_make_const_byte_span(bytes.data(),bytes.size());
  HYBRID_RETURN_IF_ERROR(iree_hal_executable_load(iree_hal_queue_family(nq),selected.target,&params,&ne));
  HYBRID_RETURN_IF_ERROR(iree_hal_executable_lookup_function_by_name(ne,IREE_SV("prefill_i8"),&nf));
  fprintf(stderr,"HYBRID_INIT npu_rows=%zu effective_fraction=%.8f banks=2 bank_rows=512\n",NN,double(NN)/N);fflush(stderr);
  return iree_ok_status();
}

iree_status_t split(const Op& op,iree_hal_buffer_binding_table_t table) {
  using Clock=std::chrono::steady_clock;const auto start=Clock::now();
  HYBRID_RETURN_IF_ERROR(initialize());
  std::array<iree_hal_buffer_ref_t,4> refs;
  for(size_t i=0;i<4;++i){refs[i]=resolve(op.refs[i],table);if(!refs[i].buffer) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,"unresolved hybrid binding");}
  std::vector<_Float16> x(M*K);std::vector<float> scales(M*2);
  HYBRID_RETURN_IF_ERROR(transfer(refs[0],x.data(),x.size()*2,false));
  HYBRID_RETURN_IF_ERROR(transfer(refs[1],scales.data(),scales.size()*4,false));
  auto key=std::make_pair(refs[2].buffer,size_t(refs[2].offset));
  auto it=weights.find(key);
  if(it==weights.end()) {
    std::vector<uint8_t> raw(NN*K/256*144);
    HYBRID_RETURN_IF_ERROR(transfer(refs[2],raw.data(),raw.size(),false));
    Weight w;w.packed.resize(NN*K);w.residual.resize(NN*K);w.scales.resize(NN);w.residual_scales.resize(NN);std::vector<float> row(K);
    for(size_t n=0;n<NN;++n) {
      float maximum=0;
      for(size_t k=0;k<K;++k){row[k]=dequant(raw.data()+(n*(K/256)+k/256)*144,k%256);maximum=std::max(maximum,std::abs(row[k]));}
      if(!std::isfinite(maximum)) return iree_make_status(IREE_STATUS_DATA_LOSS,"nonfinite weight");
      w.scales[n]=maximum?maximum/127:1;
      float residual_max=0;
      for(size_t k=0;k<K;++k) {auto q=int8_t(std::clamp(std::nearbyint(row[k]/w.scales[n]),-127.f,127.f));w.packed[bindex(n,k)]=q;row[k]-=q*w.scales[n];residual_max=std::max(residual_max,std::abs(row[k]));}
      w.residual_scales[n]=residual_max?residual_max/127:1;
      for(size_t k=0;k<K;++k)w.residual[bindex(n,k)]=int8_t(std::clamp(std::nearbyint(row[k]/w.residual_scales[n]),-127.f,127.f));
    }
    it=weights.emplace(key,std::move(w)).first;
  }
  auto& w=it->second;
  std::vector<int8_t> packed(M*K),residual(M*K);std::vector<float> xscale(M),rscale(M);
  for(size_t m=0;m<M;++m) {
    float maximum=0;for(size_t k=0;k<K;++k)maximum=std::max(maximum,std::abs(float(x[m*K+k])));
    if(!std::isfinite(maximum)) return iree_make_status(IREE_STATUS_DATA_LOSS,"nonfinite activation");
    xscale[m]=maximum?maximum/127:1;
    float residual_max=0;
    for(size_t k=0;k<K;++k){auto q=int8_t(std::clamp(std::nearbyint(float(x[m*K+k])/xscale[m]),-127.f,127.f));packed[aindex(m,k)]=q;residual_max=std::max(residual_max,std::abs(float(x[m*K+k])-q*xscale[m]));}
    rscale[m]=residual_max?residual_max/127:1;
    for(size_t k=0;k<K;++k)residual[aindex(m,k)]=int8_t(std::clamp(std::nearbyint((float(x[m*K+k])-packed[aindex(m,k)]*xscale[m])/rscale[m]),-127.f,127.f));
  }
  HYBRID_RETURN_IF_ERROR(transfer(iree_hal_make_buffer_ref(scratch,XN,packed.size()),packed.data(),packed.size(),true));
  HYBRID_RETURN_IF_ERROR(transfer(iree_hal_make_buffer_ref(scratch,WN,w.packed.size()),w.packed.data(),w.packed.size(),true));
  const auto prepared=Clock::now();uint64_t uploaded=gv;
  auto gpu_bank=[&](size_t bank)->iree_status_t {
    auto b=refs;b[0].offset+=bank*BANK*K*2;b[0].length=BANK*K*2;
    b[1].offset+=bank*BANK*8;b[1].length=BANK*8;
    b[2].offset+=NN*K/256*144;b[2].length=GN*K/256*144;
    b[3]=iree_hal_make_buffer_ref(scratch,YG+bank*BANK*GN*4,BANK*GN*4);
    uint32_t constants[3]={BANK,K,uint32_t(GN)};
    auto config=iree_hal_make_static_dispatch_config(GN/128,BANK/256,1);config.workgroup_size[0]=256;config.workgroup_size[1]=config.workgroup_size[2]=1;
    uint64_t before=gv,after=++gv;
    return iree_hal_queue_dispatch(gq,{1,&gs,&before},{1,&gs,&after},op.executable,op.function,config,
      iree_make_const_byte_span(constants,sizeof(constants)),{4,b.data()},nullptr,IREE_HAL_DISPATCH_FLAG_NONE);
  };
  auto npu_bank=[&](size_t bank)->iree_status_t {
    iree_hal_buffer_ref_t b[]={iree_hal_make_buffer_ref(scratch,XN+bank*BANK*K,BANK*K),
      iree_hal_make_buffer_ref(scratch,WN,NN*K),iree_hal_make_buffer_ref(scratch,YN+bank*BANK*NN*4,BANK*NN*4)};
    auto config=iree_hal_make_static_dispatch_config(1,1,1);config.workgroup_size[0]=config.workgroup_size[1]=config.workgroup_size[2]=1;
    iree_hal_semaphore_t* sems[]={gs,ns};uint64_t before[]={uploaded,nv},after=++nv;
    return iree_hal_queue_dispatch(nq,{2,sems,before},{1,&ns,&after},ne,nf,config,iree_const_byte_span_empty(),{3,b},nullptr,IREE_HAL_DISPATCH_FLAG_NONE);
  };
  HYBRID_RETURN_IF_ERROR(gpu_bank(0));HYBRID_RETURN_IF_ERROR(npu_bank(1));
  HYBRID_RETURN_IF_ERROR(gpu_bank(1));HYBRID_RETURN_IF_ERROR(npu_bank(0));
  HYBRID_RETURN_IF_ERROR(wait(gs,gv));HYBRID_RETURN_IF_ERROR(wait(ns,nv));
  const auto computed=Clock::now();
  std::vector<float> gy(M*GN),y(M*N);std::vector<int32_t> ny(M*NN);
  HYBRID_RETURN_IF_ERROR(transfer(iree_hal_make_buffer_ref(scratch,YG,gy.size()*4),gy.data(),gy.size()*4,false));
  HYBRID_RETURN_IF_ERROR(transfer(iree_hal_make_buffer_ref(scratch,YN,ny.size()*4),ny.data(),ny.size()*4,false));
  for(size_t m=0;m<M;++m) {
    for(size_t n=0;n<NN;++n)y[m*N+n]=float(ny[(m/BANK)*BANK*NN+cindex(m%BANK,n)])*xscale[m]*w.scales[n]*scales[m*2+1];
    memcpy(y.data()+m*N+NN,gy.data()+m*GN,GN*4);
  }
  if(getenv("LOOM_HYBRID_VALIDATE")) {
    for(size_t m: {size_t(0),size_t(17),size_t(511),size_t(512),size_t(1000)})for(size_t n: {size_t(0),size_t(7),size_t(129),NN-1}) {
      int32_t expected=0;for(size_t k=0;k<K;++k)expected+=int32_t(packed[aindex(m,k)])*w.packed[bindex(n,k)];
      if(ny[(m/BANK)*BANK*NN+cindex(m%BANK,n)]!=expected)return iree_make_status(IREE_STATUS_DATA_LOSS,"real-activation integer oracle mismatch");
    }
  }
  const char* precision=getenv("LOOM_HYBRID_PRECISION");
  if(!precision||std::string(precision)=="i8x2") {
    for(int pass=1;pass<4;++pass) {
      auto& xp=(pass&1)?residual:packed;auto& wp=(pass&2)?w.residual:w.packed;
      auto& xs=(pass&1)?rscale:xscale;auto& ws=(pass&2)?w.residual_scales:w.scales;
      HYBRID_RETURN_IF_ERROR(transfer(iree_hal_make_buffer_ref(scratch,XN,xp.size()),xp.data(),xp.size(),true));
      HYBRID_RETURN_IF_ERROR(transfer(iree_hal_make_buffer_ref(scratch,WN,wp.size()),wp.data(),wp.size(),true));
      uploaded=gv;HYBRID_RETURN_IF_ERROR(npu_bank(1));HYBRID_RETURN_IF_ERROR(npu_bank(0));HYBRID_RETURN_IF_ERROR(wait(ns,nv));
      HYBRID_RETURN_IF_ERROR(transfer(iree_hal_make_buffer_ref(scratch,YN,ny.size()*4),ny.data(),ny.size()*4,false));
      for(size_t m=0;m<M;++m)for(size_t n=0;n<NN;++n)y[m*N+n]+=float(ny[(m/BANK)*BANK*NN+cindex(m%BANK,n)])*xs[m]*ws[n]*scales[m*2+1];
      if(getenv("LOOM_HYBRID_VALIDATE"))for(size_t m:{size_t(0),size_t(511),size_t(512),size_t(1000)})for(size_t n:{size_t(0),size_t(129),NN-1}) {
        int32_t expected=0;for(size_t k=0;k<K;++k)expected+=int32_t(xp[aindex(m,k)])*wp[bindex(n,k)];
        if(ny[(m/BANK)*BANK*NN+cindex(m%BANK,n)]!=expected)return iree_make_status(IREE_STATUS_DATA_LOSS,"residual integer oracle mismatch");
      }
    }
  }
  if(getenv("LOOM_HYBRID_GPU_ORACLE")&&split_count==0) {
    iree_hal_buffer_t* reference=nullptr;
    HYBRID_RETURN_IF_ERROR(iree_hal_pool_allocate_buffer(pool,iree_hal_make_pool_reservation_request(M*N*4,64),iree_make_timeout_ms(120000),&reference));
    auto b=refs;b[3]=iree_hal_make_buffer_ref(reference,0,M*N*4);
    uint32_t constants[3]={M,K,N};auto config=iree_hal_make_static_dispatch_config(N/128,M/256,1);config.workgroup_size[0]=256;config.workgroup_size[1]=config.workgroup_size[2]=1;
    uint64_t before=gv,after=++gv;
    HYBRID_RETURN_IF_ERROR(iree_hal_queue_dispatch(gq,{1,&gs,&before},{1,&gs,&after},op.executable,op.function,config,iree_make_const_byte_span(constants,12),{4,b.data()},nullptr,IREE_HAL_DISPATCH_FLAG_NONE));
    HYBRID_RETURN_IF_ERROR(wait(gs,after));std::vector<float> expected(M*N);
    HYBRID_RETURN_IF_ERROR(transfer(b[3],expected.data(),expected.size()*4,false));
    double err[2]={},norm[2]={},maximum[2]={};size_t mismatch[2]={};
    for(size_t m=0;m<M;++m)for(size_t n=0;n<N;++n){size_t i=m*N+n,side=n<NN?0:1;double delta=double(y[i])-expected[i];err[side]+=delta*delta;norm[side]+=double(expected[i])*expected[i];maximum[side]=std::max(maximum[side],std::abs(delta));mismatch[side]+=y[i]!=expected[i];}
    fprintf(stderr,"HYBRID_GPU_ORACLE npu_rel_l2=%.9g gpu_rel_l2=%.9g npu_max_abs=%.9g gpu_max_abs=%.9g npu_mismatches=%zu gpu_mismatches=%zu\n",sqrt(err[0]/norm[0]),sqrt(err[1]/norm[1]),maximum[0],maximum[1],mismatch[0],mismatch[1]);fflush(stderr);
    iree_hal_buffer_release(reference);
  }
  HYBRID_RETURN_IF_ERROR(transfer(refs[3],y.data(),y.size()*4,true));
  const auto end=Clock::now();++split_count;
  fprintf(stderr,"HYBRID_SPLIT count=%zu prepare_ms=%.3f compute_ms=%.3f assemble_ms=%.3f total_ms=%.3f\n",split_count,
      std::chrono::duration<double,std::milli>(prepared-start).count(),std::chrono::duration<double,std::milli>(computed-prepared).count(),
      std::chrono::duration<double,std::milli>(end-computed).count(),std::chrono::duration<double,std::milli>(end-start).count());fflush(stderr);
  if(getenv("LOOM_HYBRID_PROFILE") && profile_state==1 && split_count%16==0) {
    HYBRID_RETURN_IF_ERROR(iree_hal_device_profiling_flush(gpu));
    HYBRID_RETURN_IF_ERROR(iree_hal_device_profiling_flush(npu));
  }
  return iree_ok_status();
}
}

extern "C" iree_status_t iree_hal_driver_create_device_by_uri(iree_hal_driver_t* driver,iree_string_view_t uri,
  const iree_hal_device_create_params_t* params,iree_allocator_t allocator,iree_hal_device_t** out) {
  static auto real=next<decltype(&iree_hal_driver_create_device_by_uri)>("iree_hal_driver_create_device_by_uri");
  HYBRID_RETURN_IF_ERROR(real(driver,uri,params,allocator,out));
  fprintf(stderr,"HYBRID_DEVICE uri=%.*s\n",int(uri.size),uri.data);fflush(stderr);
  if(!internal&&!gpu&&(std::string(uri.data,uri.size).find("amdgpu")==0||std::string(uri.data,uri.size).find("rdna")==0)) {
    gpu=*out;iree_hal_device_retain(gpu);Guard guard;
    HYBRID_RETURN_IF_ERROR(real(driver,IREE_SV("xdna://0?columns=8"),params,allocator,&npu));
  }
  return iree_ok_status();
}
extern "C" iree_status_t iree_hal_device_group_builder_finalize(iree_hal_device_group_builder_t* b,iree_allocator_t a,iree_hal_device_group_t** out) {
  static auto real=next<decltype(&iree_hal_device_group_builder_finalize)>("iree_hal_device_group_builder_finalize");
  fprintf(stderr,"HYBRID_GROUP count=%zu gpu=%p npu=%p\n",b->count,(void*)gpu,(void*)npu);fflush(stderr);
  bool match=false;for(size_t i=0;i<b->count;++i)match|=b->devices[i]==gpu;
  if(match&&npu&&!group) {HYBRID_RETURN_IF_ERROR(iree_hal_device_group_builder_add_device(b,npu));
    HYBRID_RETURN_IF_ERROR(real(b,a,out));group=*out;iree_hal_device_group_retain(group);return iree_ok_status();}
  return real(b,a,out);
}
extern "C" iree_status_t iree_hal_command_buffer_create(
    const iree_hal_queue_family_t* family, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t categories, iree_host_size_t capacity,
    iree_hal_command_buffer_t** out) {
  static auto real=next<decltype(&iree_hal_command_buffer_create)>("iree_hal_command_buffer_create");
  if(getenv("LOOM_HYBRID_PROFILE"))mode|=IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA;
  return real(family,mode,categories,capacity,out);
}
extern "C" iree_status_t iree_hal_command_buffer_begin(iree_hal_command_buffer_t* cb) {
  static auto real=next<decltype(&iree_hal_command_buffer_begin)>("iree_hal_command_buffer_begin");
  HYBRID_RETURN_IF_ERROR(real(cb));
  if(!internal){std::lock_guard<std::mutex> lock(records_mutex);records[cb]=std::make_shared<Recording>();}
  return iree_ok_status();
}
extern "C" iree_status_t iree_hal_command_buffer_dispatch(iree_hal_command_buffer_t* cb,iree_hal_executable_t* e,
  iree_hal_executable_function_t f,iree_hal_dispatch_config_t config,iree_const_byte_span_t constants,iree_hal_buffer_ref_list_t bindings,iree_hal_dispatch_flags_t flags) {
  static auto real=next<decltype(&iree_hal_command_buffer_dispatch)>("iree_hal_command_buffer_dispatch");
  HYBRID_RETURN_IF_ERROR(real(cb,e,f,config,constants,bindings,flags));if(internal)return iree_ok_status();
  std::vector<uint8_t> c(constants.data,constants.data+constants.data_length);
  std::vector<iree_hal_buffer_ref_t> b(bindings.values,bindings.values+bindings.count);
  Op op;op.replay=[=](auto target){return real(target,e,f,config,iree_make_const_byte_span(c.data(),c.size()),{b.size(),b.data()},flags);};
  iree_hal_executable_function_info_t info={};HYBRID_RETURN_IF_ERROR(iree_hal_executable_function_info(e,f,&info));
  const char* pattern=getenv("LOOM_HYBRID_ROOT");const std::string root=pattern?pattern:"g_gemm_q4_k_store_m256";
  if(std::string(info.name.data,info.name.size)==root&&c.size()==12&&b.size()==4&&!flags) {
    uint32_t dims[3];memcpy(dims,c.data(),12);
    if(dims[0]==M&&dims[1]==K&&dims[2]==N){op.split=true;op.executable=e;op.function=f;std::copy(b.begin(),b.end(),op.refs.begin());}
  }
  append(cb,std::move(op));return iree_ok_status();
}
extern "C" iree_status_t iree_hal_command_buffer_barrier(iree_hal_command_buffer_t* cb,const iree_hal_barrier_t* b) {
  static auto real=next<decltype(&iree_hal_command_buffer_barrier)>("iree_hal_command_buffer_barrier");
  HYBRID_RETURN_IF_ERROR(real(cb,b));if(internal)return iree_ok_status();
  auto value=*b;
  std::vector<iree_hal_memory_barrier_t> memory(b->memory_barriers,b->memory_barriers+b->memory_barrier_count);
  std::vector<iree_hal_buffer_barrier_t> buffers(b->buffer_barriers,b->buffer_barriers+b->buffer_barrier_count);
  append(cb,{[=](auto target)mutable{value.memory_barriers=memory.data();value.buffer_barriers=buffers.data();return real(target,&value);}});
  return iree_ok_status();
}
extern "C" iree_status_t iree_hal_command_buffer_copy_buffer(iree_hal_command_buffer_t* cb,iree_hal_buffer_ref_t source,iree_hal_buffer_ref_t target,iree_hal_copy_flags_t flags) {
  static auto real=next<decltype(&iree_hal_command_buffer_copy_buffer)>("iree_hal_command_buffer_copy_buffer");
  HYBRID_RETURN_IF_ERROR(real(cb,source,target,flags));append(cb,{[=](auto c){return real(c,source,target,flags);}});return iree_ok_status();
}
extern "C" iree_status_t iree_hal_command_buffer_fill_buffer(iree_hal_command_buffer_t* cb,iree_hal_buffer_ref_t target,const void* p,iree_host_size_t length,iree_hal_fill_flags_t flags) {
  static auto real=next<decltype(&iree_hal_command_buffer_fill_buffer)>("iree_hal_command_buffer_fill_buffer");
  HYBRID_RETURN_IF_ERROR(real(cb,target,p,length,flags));uint32_t pattern=0;memcpy(&pattern,p,length);
  append(cb,{[=](auto c){return real(c,target,&pattern,length,flags);}});return iree_ok_status();
}
static iree_status_t execute_hybrid(iree_hal_queue_t* queue,iree_hal_semaphore_list_t waits,iree_hal_semaphore_list_t signals,
  iree_hal_command_buffer_t* cb,iree_hal_buffer_binding_table_t table,iree_hal_queue_execute_flags_t flags) {
  static auto real=next<decltype(&iree_hal_queue_execute)>("iree_hal_queue_execute");
  if(internal)return real(queue,waits,signals,cb,table,flags);
  std::shared_ptr<Recording> recording;
  {std::lock_guard<std::mutex> lock(records_mutex);auto it=records.find(cb);if(it!=records.end())recording=it->second;}
  if(!recording||!recording->split)return real(queue,waits,signals,cb,table,flags);
  for(size_t i=0;i<waits.count;++i){HYBRID_RETURN_IF_ERROR(wait(waits.semaphores[i],waits.payload_values[i]));}
  std::lock_guard<std::mutex> lock(run_mutex);Guard guard;
  HYBRID_RETURN_IF_ERROR(initialize());
  iree_hal_command_buffer_t* segment=nullptr;size_t count=0;
  auto begin=[&]()->iree_status_t {
    HYBRID_RETURN_IF_ERROR(iree_hal_command_buffer_create(iree_hal_queue_family(queue),iree_hal_command_buffer_mode(cb),
       iree_hal_command_buffer_allowed_categories(cb),table.count,&segment));
    return iree_hal_command_buffer_begin(segment);
  };
  auto flush=[&]()->iree_status_t {
    if(!segment)return iree_ok_status();
    iree_status_t status=iree_hal_command_buffer_end(segment);
    if(iree_status_is_ok(status)&&count){uint64_t before=gv,after=++gv;status=real(queue,{1,&gs,&before},{1,&gs,&after},segment,table,flags);if(iree_status_is_ok(status))status=wait(gs,after);}
    iree_hal_command_buffer_release(segment);segment=nullptr;count=0;return status;
  };
  iree_status_t status=iree_ok_status();
  for(const auto& op:recording->ops) {
    if(op.split){status=flush();if(iree_status_is_ok(status))status=split(op,table);}
    else {if(!segment)status=begin();if(iree_status_is_ok(status)){status=op.replay(segment);++count;}}
    if(!iree_status_is_ok(status))break;
  }
  if(iree_status_is_ok(status))status=flush();else if(segment)iree_hal_command_buffer_release(segment);
  if(iree_status_is_ok(status))status=iree_hal_queue_barrier(queue,{1,&gs,&gv},signals,nullptr,IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  if(!iree_status_is_ok(status))iree_status_fprint(stderr,status);
  return status;
}

extern "C" iree_status_t iree_hal_queue_execute(iree_hal_queue_t* queue,
    iree_hal_semaphore_list_t waits,iree_hal_semaphore_list_t signals,
    iree_hal_command_buffer_t* cb,iree_hal_buffer_binding_table_t table,
    iree_hal_queue_execute_flags_t flags) {
  static auto real=next<decltype(&iree_hal_queue_execute)>("iree_hal_queue_execute");
  if(internal)return real(queue,waits,signals,cb,table,flags);
  HYBRID_RETURN_IF_ERROR(profile_control());
  bool selected=false;
  {std::lock_guard<std::mutex> lock(records_mutex);auto it=records.find(cb);selected=it!=records.end()&&it->second->split;}
  if(!selected)return real(queue,waits,signals,cb,table,flags);
  std::vector<iree_hal_semaphore_t*> ws,ss;
  std::vector<uint64_t> wv,sv;
  std::vector<iree_hal_buffer_binding_t> bindings;
  for(size_t i=0;i<waits.count;++i){ws.push_back(waits.semaphores[i]);wv.push_back(waits.payload_values[i]);iree_hal_semaphore_retain(ws.back());}
  for(size_t i=0;i<signals.count;++i){ss.push_back(signals.semaphores[i]);sv.push_back(signals.payload_values[i]);iree_hal_semaphore_retain(ss.back());}
  for(size_t i=0;i<table.count;++i){bindings.push_back(table.bindings[i]);iree_hal_buffer_retain(bindings.back().buffer);}
  iree_hal_command_buffer_retain(cb);iree_hal_queue_retain(queue);
  std::thread([=]()mutable{
    auto status=execute_hybrid(queue,{ws.size(),ws.data(),wv.data()},{ss.size(),ss.data(),sv.data()},cb,{bindings.size(),bindings.data()},flags);
    if(!iree_status_is_ok(status)){
      iree_status_fprint(stderr,status);fflush(stderr);
      for(auto* semaphore:ss)iree_hal_semaphore_fail(semaphore,iree_status_clone(status));
      iree_status_ignore(status);
    }
    for(auto* semaphore:ws)iree_hal_semaphore_release(semaphore);
    for(auto* semaphore:ss)iree_hal_semaphore_release(semaphore);
    for(auto& binding:bindings)iree_hal_buffer_release(binding.buffer);
    iree_hal_command_buffer_release(cb);iree_hal_queue_release(queue);
  }).detach();
  return iree_ok_status();
}

#ifdef LOOM_HYBRID_UNIT_TEST
extern "C" void hybrid_dequant_block(const uint8_t* block,float* output) {
  for(size_t k=0;k<256;++k)output[k]=dequant(block,k);
}
extern "C" size_t hybrid_packed_a(size_t row,size_t col) {return aindex(row,col);}
extern "C" size_t hybrid_packed_b(size_t row,size_t col) {return bindex(row,col);}
#endif
