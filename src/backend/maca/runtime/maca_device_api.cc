/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file maca_device_api.cc
 * \brief GPU specific API
 */
#include <mcc/mcc_global.h>
#include <mcr/mc_runtime_api.h>
#include <mxc/mxc.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/timer.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "maca_common.h"
#include "tvm/ffi/base_details.h"

namespace tvm {
namespace runtime {

class MACADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final { MACA_CALL(mcSetDevice(dev.device_id)); }
  ffi::Any GetAttr(Device device, const ffi::String& name) final {
    ffi::Any rv;
    int value = 0;
    if (name == "exist") {
      int count;
      auto err = mcGetDeviceCount(&count);
      return err == mcSuccess && device.device_id >= 0 && device.device_id < count;
    }
    if (name == "max_threads_per_block") {
      MACA_CALL(
          mcDeviceGetAttribute(&value, mcDeviceAttributeMaxThreadsPerBlock, device.device_id));
      return value;
    }
    if (name == "warp_size") {
      MACA_CALL(mcDeviceGetAttribute(&value, mcDeviceAttributeWarpSize, device.device_id));
      return value;
    }
    if (name == "max_shared_memory_per_block") {
      MACA_CALL(
          mcDeviceGetAttribute(&value, mcDeviceAttributeMaxSharedMemoryPerBlock, device.device_id));
      return value;
    }
    if (name == "compute_version") {
      std::ostringstream os;
      MACA_CALL(
          mcDeviceGetAttribute(&value, mcDeviceAttributeComputeCapabilityMajor, device.device_id));
      os << value << ".";
      MACA_CALL(
          mcDeviceGetAttribute(&value, mcDeviceAttributeComputeCapabilityMinor, device.device_id));
      os << std::setfill('0') << std::setw(2) << value;
      return os.str();
    }
    if (name == "device_name") {
      std::string name(256, 0);
      MACA_CALL(mcDeviceGetName(&name[0], name.size(), device.device_id));
      name.resize(strlen(name.c_str()));
      return std::move(name);
    }
    if (name == "max_clock_rate") {
      MACA_CALL(mcDeviceGetAttribute(&value, mcDeviceAttributeClockRate, device.device_id));
      return value;
    }
    if (name == "multi_processor_count") {
      MACA_CALL(
          mcDeviceGetAttribute(&value, mcDeviceAttributeMultiProcessorCount, device.device_id));
      return value;
    }
    if (name == "max_thread_dimensions") {
      int dims[3];
      MACA_CALL(mcDeviceGetAttribute(&dims[0], mcDeviceAttributeMaxBlockDimX, device.device_id));
      MACA_CALL(mcDeviceGetAttribute(&dims[1], mcDeviceAttributeMaxBlockDimY, device.device_id));
      MACA_CALL(mcDeviceGetAttribute(&dims[2], mcDeviceAttributeMaxBlockDimZ, device.device_id));

      std::stringstream ss;
      ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
      return ss.str();
    }
    if (name == "max_registers_per_block") {
      MACA_CALL(
          mcDeviceGetAttribute(&value, mcDeviceAttributeMaxRegistersPerBlock, device.device_id));
      return value;
    }
    if (name == "l2_cache_size_bytes") {
      // Get size of device l2 cache size in bytes.
      int l2_size;
      MACA_CALL(mcDeviceGetAttribute(&l2_size, mcDeviceAttributeL2CacheSize, device.device_id));
      return l2_size;
    }
    if (name == "total_global_memory") {
      mcDeviceProp_t prop;
      MACA_CALL(mcGetDeviceProperties(&prop, device.device_id));
      int64_t total_global_memory = prop.totalGlobalMem;
      return total_global_memory;
    }
    if (name == "available_global_memory") {
      size_t free_mem, total_mem;
      MACA_CALL(mcMemGetInfo(&free_mem, &total_mem));
      return static_cast<int64_t>(free_mem);
    }
    return rv;
  }
  ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, Device device,
                    ffi::Optional<ffi::String> mem_scope = std::nullopt) final {
    TVM_FFI_CHECK(
        !mem_scope.has_value() || mem_scope.value().empty() || mem_scope.value() == "global",
        ValueError)
        << "Unsupported memory scope: " << mem_scope.value_or("");
    struct Allocator {
      MACADeviceAPI* api;
      void AllocData(DLTensor* tensor) {
        size_t alignment =
            std::max(static_cast<size_t>(kAllocAlignment),
                     static_cast<size_t>((tensor->dtype.bits / 8) * tensor->dtype.lanes));
        tensor->data = api->AllocDataSpace(tensor->device, ffi::GetDataSize(*tensor), alignment,
                                           tensor->dtype);
      }
      void FreeData(DLTensor* tensor) { api->FreeDataSpace(tensor->device, tensor->data); }
    };
    return ffi::Tensor::FromNDAlloc(Allocator{this}, shape, dtype, device);
  }

 private:
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) {
    TVM_FFI_ICHECK_EQ(256 % alignment, 0U) << "MACA space is aligned at 256 bytes";
    void* ret;
    if (dev.device_type == kDLMACAHost) {
      VLOG(1) << "allocating " << nbytes << "bytes on host";
      MACA_CALL(mcMallocHost(&ret, nbytes));
    } else {
      MACA_CALL(mcSetDevice(dev.device_id));
      VLOG(1) << "allocating " << nbytes << " bytes on device";
      MACA_CALL(mcMalloc(&ret, nbytes));
    }
    return ret;
  }

  void FreeDataSpace(Device dev, void* ptr) {
    if (dev.device_type == kDLMACAHost) {
      MACA_CALL(mcFreeHost(ptr));
      return;
    }

    MACADeviceGuard guard(dev.device_id);
    MACA_DRIVER_CALL(mcFree(ptr));
  }

 public:
 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    mcStream_t maca_stream = static_cast<mcStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;
    if (dev_from.device_type == kDLMACAHost) {
      dev_from.device_type = kDLCPU;
    }
    if (dev_to.device_type == kDLMACAHost) {
      dev_to.device_type = kDLCPU;
    }

    if (dev_from.device_type == kDLMACA && dev_to.device_type == kDLMACA) {
      MACA_CALL(mcSetDevice(dev_from.device_id));
      if (dev_from.device_id == dev_to.device_id) {
        GPUCopy(from, to, size, mcMemcpyDeviceToDevice, maca_stream);
      } else {
        MACA_CALL(
            mcMemcpyPeerAsync(to, dev_to.device_id, from, dev_from.device_id, size, maca_stream));
      }
    } else if (dev_from.device_type == kDLMACA && dev_to.device_type == kDLCPU) {
      MACA_CALL(mcSetDevice(dev_from.device_id));
      GPUCopy(from, to, size, mcMemcpyDeviceToHost, maca_stream);
    } else if (dev_from.device_type == kDLCPU && dev_to.device_type == kDLMACA) {
      MACA_CALL(mcSetDevice(dev_to.device_id));
      GPUCopy(from, to, size, mcMemcpyHostToDevice, maca_stream);
    } else {
      TVM_FFI_THROW(InternalError) << "expect copy from/to GPU or between GPU";
    }
  }

 public:
  TVMStreamHandle CreateStream(Device dev) {
    MACA_CALL(mcSetDevice(dev.device_id));
    mcStream_t retval;
    MACA_CALL(mcStreamCreateWithFlags(&retval, mcStreamNonBlocking));
    return static_cast<TVMStreamHandle>(retval);
  }

  void FreeStream(Device dev, TVMStreamHandle stream) {
    MACADeviceGuard guard(dev.device_id);
    MACA_DRIVER_CALL(mcStreamDestroy(static_cast<mcStream_t>(stream)));
  }

  void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    MACA_CALL(mcSetDevice(dev.device_id));
    mcStream_t src_stream = static_cast<mcStream_t>(event_src);
    mcStream_t dst_stream = static_cast<mcStream_t>(event_dst);
    mcEvent_t evt;
    MACA_CALL(mcEventCreate(&evt));
    MACA_CALL(mcEventRecord(evt, src_stream));
    MACA_CALL(mcStreamWaitEvent(dst_stream, evt, 0));
    MACA_CALL(mcEventDestroy(evt));
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    MACA_CALL(mcSetDevice(dev.device_id));
    MACA_CALL(mcStreamSynchronize(static_cast<mcStream_t>(stream)));
  }

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final {
    return MACAThreadEntry::ThreadLocal()->pool.AllocWorkspace(dev, size);
  }

  void FreeWorkspace(Device dev, void* data) final {
    MACAThreadEntry::ThreadLocal()->pool.FreeWorkspace(dev, data);
  }

  static MACADeviceAPI* Global() {
    static MACADeviceAPI* inst = new MACADeviceAPI();
    return inst;
  }

 private:
  static void GPUCopy(const void* from, void* to, size_t size, mcMemcpyKind kind,
                      mcStream_t stream) {
    MACA_CALL(mcMemcpyAsync(to, from, size, kind, stream));
  }
};

MACAThreadEntry::MACAThreadEntry() : pool(kDLMACA, MACADeviceAPI::Global()) {}

MACAThreadEntry* MACAThreadEntry::ThreadLocal() {
  static thread_local MACAThreadEntry inst;
  return &inst;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("device_api.maca",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    DeviceAPI* ptr = MACADeviceAPI::Global();
                    *rv = static_cast<void*>(ptr);
                  })
      .def_packed("device_api.maca_host", [](ffi::PackedArgs args, ffi::Any* rv) {
        DeviceAPI* ptr = MACADeviceAPI::Global();
        *rv = static_cast<void*>(ptr);
      });
}

class MACATimerObj : public TimerObj {
 public:
  virtual void Start() {
    int device_id;
    MACA_CALL(mcGetDevice(&device_id));
    stream_ = TVMFFIEnvGetStream(kDLMACA, device_id);
    MACA_CALL(mcEventRecord(start_, static_cast<mcStream_t>(stream_)));
  }
  virtual void Stop() { MACA_CALL(mcEventRecord(stop_, static_cast<mcStream_t>(stream_))); }
  virtual int64_t SyncAndGetElapsedNanos() {
    MACA_CALL(mcEventSynchronize(stop_));
    float milliseconds = 0;
    MACA_CALL(mcEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds * 1e6;
  }
  virtual ~MACATimerObj() {
    MACA_CALL(mcEventDestroy(start_));
    MACA_CALL(mcEventDestroy(stop_));
  }
  MACATimerObj() {
    MACA_CALL(mcEventCreate(&start_));
    MACA_CALL(mcEventCreate(&stop_));
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.maca.MACATimerNode", MACATimerObj, TimerObj);

 private:
  mcEvent_t start_;
  mcEvent_t stop_;
  TVMStreamHandle stream_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.timer.maca", [](Device dev) {
    return Timer(ffi::UnsafeInit{}, ffi::make_object<MACATimerObj>());
  });
}
}  // namespace runtime
}  // namespace tvm
