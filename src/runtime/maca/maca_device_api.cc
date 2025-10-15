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
#include <dmlc/thread_local.h>
#include <mcr/mc_runtime_api.h>
#include <mxc/mxc.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include "maca_common.h"

namespace tvm {
namespace runtime {

class MACADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final { MACA_CALL(mcSetDevice(dev.device_id)); }
  void GetAttr(Device device, DeviceAttrKind kind, TVMRetValue* rv) final {
    int value = 0;
    switch (kind) {
      case kExist: {
        if (mxc_init() == MXC_STATUS_SUCCESS) {
          int dev;
          MACA_CALL(mcGetDeviceCount(&dev));
          value = dev > device.device_id ? 1 : 0;
          mxc_shut_down();
        } else {
          value = 0;
        }
        break;
      }
      case kMaxThreadsPerBlock: {
        MACA_CALL(
            mcDeviceGetAttribute(&value, mcDeviceAttributeMaxThreadsPerBlock, device.device_id));
        break;
      }
      case kWarpSize: {
        MACA_CALL(mcDeviceGetAttribute(&value, mcDeviceAttributeWarpSize, device.device_id));
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        MACA_CALL(mcDeviceGetAttribute(&value, mcDeviceAttributeMaxSharedMemoryPerBlock,
                                       device.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        MACA_CALL(mcDeviceGetAttribute(&value, mcDeviceAttributeComputeCapabilityMajor,
                                       device.device_id));
        os << value << ".";
        MACA_CALL(mcDeviceGetAttribute(&value, mcDeviceAttributeComputeCapabilityMinor,
                                       device.device_id));
        os << value;
        *rv = os.str();
        return;
      }
      case kDeviceName: {
        std::string name(256, 0);
        MACA_CALL(mcDeviceGetName(&name[0], name.size(), device.device_id));
        name.resize(strlen(name.c_str()));
        *rv = std::move(name);
        return;
      }
      case kMaxClockRate: {
        MACA_CALL(mcDeviceGetAttribute(&value, mcDeviceAttributeClockRate, device.device_id));
        break;
      }
      case kMultiProcessorCount: {
        MACA_CALL(
            mcDeviceGetAttribute(&value, mcDeviceAttributeMultiProcessorCount, device.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        MACA_CALL(mcDeviceGetAttribute(&dims[0], mcDeviceAttributeMaxBlockDimX, device.device_id));
        MACA_CALL(mcDeviceGetAttribute(&dims[1], mcDeviceAttributeMaxBlockDimY, device.device_id));
        MACA_CALL(mcDeviceGetAttribute(&dims[2], mcDeviceAttributeMaxBlockDimZ, device.device_id));

        std::stringstream ss;
        ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
        *rv = ss.str();
        return;
      }
      case kMaxRegistersPerBlock:
        MACA_CALL(
            mcDeviceGetAttribute(&value, mcDeviceAttributeMaxRegistersPerBlock, device.device_id));
        break;
      case kMxcArch: {
        mcDeviceProp_t prop;
        MACA_CALL(mcGetDeviceProperties(&prop, device.device_id));
        *rv = prop.mxArchName;
        return;
      }
      case kApiVersion: {
        //*rv = MACA_VERSION;
        return;
      }
      case kDriverVersion:
        return;
      case kL2CacheSizeBytes: {
        // Get size of device l2 cache size in bytes.
        int l2_size;
        MACA_CALL(mcDeviceGetAttribute(&l2_size, mcDeviceAttributeL2CacheSize, device.device_id));
        *rv = l2_size;
        return;
      }
      case kTotalGlobalMemory: {
        mcDeviceProp_t prop;
        MACA_CALL(mcGetDeviceProperties(&prop, device.device_id));
        int64_t total_global_memory = prop.totalGlobalMem;
        *rv = total_global_memory;
        return;
      }
      case kAvailableGlobalMemory: {
        size_t free_mem, total_mem;
        MACA_CALL(mcMemGetInfo(&free_mem, &total_mem));
        *rv = static_cast<int64_t>(free_mem);
        return;
      }
    }
    *rv = value;
  }
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    ICHECK_EQ(256 % alignment, 0U) << "MACA space is aligned at 256 bytes";
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

  void FreeDataSpace(Device dev, void* ptr) final {
    if (dev.device_type == kDLMACAHost) {
      MACA_CALL(mcFreeHost(ptr));
    } else {
      MACA_DRIVER_CALL(mcSetDevice(dev.device_id));
      MACA_DRIVER_CALL(mcFree(ptr));
    }
  }

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
      LOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    MACA_CALL(mcSetDevice(dev.device_id));
    MACA_CALL(mcStreamSynchronize(static_cast<mcStream_t>(stream)));
  }

  void SetStream(Device dev, TVMStreamHandle stream) final {
    MACAThreadEntry::ThreadLocal()->stream = static_cast<mcStream_t>(stream);
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
    if (stream != 0) {
      MACA_CALL(mcMemcpyAsync(to, from, size, kind, stream));
    } else {
      MACA_CALL(mcMemcpy(to, from, size, kind));
    }
  }
};

typedef dmlc::ThreadLocalStore<MACAThreadEntry> MACAThreadStore;

MACAThreadEntry::MACAThreadEntry() : pool(kDLMACA, MACADeviceAPI::Global()) {}

MACAThreadEntry* MACAThreadEntry::ThreadLocal() { return MACAThreadStore::Get(); }

TVM_REGISTER_GLOBAL("device_api.maca").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = MACADeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

TVM_REGISTER_GLOBAL("device_api.maca_host").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = MACADeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

class MACATimerNode : public TimerNode {
 public:
  virtual void Start() { MACA_CALL(mcEventRecord(start_, MACAThreadEntry::ThreadLocal()->stream)); }
  virtual void Stop() { MACA_CALL(mcEventRecord(stop_, MACAThreadEntry::ThreadLocal()->stream)); }
  virtual int64_t SyncAndGetElapsedNanos() {
    MACA_CALL(mcEventSynchronize(stop_));
    float milliseconds = 0;
    MACA_CALL(mcEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds * 1e6;
  }
  virtual ~MACATimerNode() {
    MACA_CALL(mcEventDestroy(start_));
    MACA_CALL(mcEventDestroy(stop_));
  }
  MACATimerNode() {
    MACA_CALL(mcEventCreate(&start_));
    MACA_CALL(mcEventCreate(&stop_));
  }

  static constexpr const char* _type_key = "MACATimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(MACATimerNode, TimerNode);

 private:
  mcEvent_t start_;
  mcEvent_t stop_;
};

TVM_REGISTER_OBJECT_TYPE(MACATimerNode);

TVM_REGISTER_GLOBAL("profiling.timer.maca").set_body_typed([](Device dev) {
  return Timer(make_object<MACATimerNode>());
});

TVM_REGISTER_GLOBAL("runtime.get_maca_stream").set_body_typed([]() {
  return static_cast<void*>(MACAThreadEntry::ThreadLocal()->stream);
});

}  // namespace runtime
}  // namespace tvm
