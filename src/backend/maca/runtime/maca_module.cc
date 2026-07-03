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
 * \file maca_module.cc
 */
#include "maca_module.h"

#include <mcr/mc_runtime_api.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <array>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../runtime/file_utils.h"
#include "../../../runtime/metadata.h"
#include "../../../runtime/pack_args.h"
#include "../../../runtime/thread_storage_scope.h"
#include "../../../support/bytes_io.h"
#include "maca_common.h"

namespace tvm {
namespace runtime {

// Module to support thread-safe multi-GPU execution.
// mcModule_t is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class MACAModuleNode : public ffi::ModuleObj {
 public:
  explicit MACAModuleNode(ffi::Bytes data, ffi::String fmt,
                          ffi::Map<ffi::String, FunctionInfo> fmap, ffi::String maca_source)
      : data_(data), fmt_(fmt), fmap_(fmap), maca_source_(maca_source) {
    std::fill(module_.begin(), module_.end(), nullptr);
  }
  // destructor
  ~MACAModuleNode() {
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        MACA_CALL(mcSetDevice(static_cast<int>(i)));
        MACA_DRIVER_CALL(mcModuleUnload(module_[i]));
      }
    }
  }

  const char* kind() const final { return "maca"; }

  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }
  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  void WriteToFile(const ffi::String& file_name, const ffi::String& format) const final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "maca") {
      TVM_FFI_ICHECK_NE(maca_source_.length(), 0);
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, std::string(maca_source_.data(), maca_source_.size()));
    } else {
      TVM_FFI_ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, std::string(data_.data(), data_.size()));
    }
  }

  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(data_);
    return ffi::Bytes(buffer);
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (format == fmt_) return ffi::String(data_.data(), data_.size());
    if (maca_source_.length() != 0) {
      return maca_source_;
    } else {
      if (fmt_ == "fatbin") return ffi::String(data_.data(), data_.size());
      return "";
    }
  }

  // get a mcFunction_t from primary context in device_id
  mcFunction_t GetFunc(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope

    if (module_[device_id] == nullptr) {
      MACA_DRIVER_CALL(mcModuleLoadData(&(module_[device_id]), data_.data()));
    }
    mcFunction_t func;
    mcError_t result = mcModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != mcSuccess) {
      TVM_FFI_THROW(InternalError) << "MACAError: mcModuleGetFunction " << func_name
                                   << " failed with error: " << mcGetErrorString(result);
    }
    return func;
  }
  // get a global var from primary context in device_id
  mcDeviceptr_t GetGlobal(int device_id, const std::string& global_name, size_t expect_nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      MACA_DRIVER_CALL(mcModuleLoadData(&(module_[device_id]), data_.data()));
    }
    mcDeviceptr_t global = nullptr;
    size_t nbytes = 0;

    MACA_DRIVER_CALL(mcModuleGetGlobal(&global, &nbytes, module_[device_id], global_name.c_str()));
    TVM_FFI_ICHECK_EQ(nbytes, expect_nbytes);
    return global;
  }

 private:
  // the binary data
  ffi::Bytes data_;
  // The format
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  // The maca source.
  ffi::String maca_source_;
  // the internal modules per GPU, to be lazily initialized.
  std::array<mcModule_t, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class MACAWrappedFunc {
 public:
  // initialize the MACA function.
  void Init(MACAModuleNode* m, ffi::ObjectPtr<ffi::Object> sptr, const std::string& func_name,
            size_t num_void_args, const ffi::Array<ffi::String>& launch_param_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }
  // invoke the function with void arguments
  void operator()(ffi::PackedArgs args, ffi::Any* rv, void* packed_args,
                  size_t packed_nbytes) const {
    int device_id;
    MACA_CALL(mcGetDevice(&device_id));
    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    }

    mcStream_t strm = static_cast<mcStream_t>(TVMFFIEnvGetStream(kDLMACA, device_id));

    ThreadWorkLoad wl = launch_param_config_.Extract(args);
    void* config[] = {MC_LAUNCH_PARAM_BUFFER_POINTER, packed_args, MC_LAUNCH_PARAM_BUFFER_SIZE,
                      &packed_nbytes, MC_LAUNCH_PARAM_END};
    // MACA supports only extra_args.
    MACA_DRIVER_CALL(mcModuleLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                          wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                          wl.block_dim(2), wl.dyn_shmem_size, strm, nullptr,
                                          reinterpret_cast<void**>(&config)));
  }

 private:
  // internal module
  MACAModuleNode* m_;
  // the resource holder
  ffi::ObjectPtr<ffi::Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<mcFunction_t, kMaxNumGPUs> fcache_;
  // launch parameters configuration
  LaunchParamConfig launch_param_config_;
};

ffi::Optional<ffi::Function> MACAModuleNode::GetFunction(const ffi::String& name) {
  ffi::ObjectPtr<ffi::Object> sptr_to_self = ffi::GetObjectPtr<ffi::Object>(this);
  TVM_FFI_ICHECK_EQ(sptr_to_self.get(), this);
  auto opt_info = fmap_.Get(name);
  if (!opt_info.has_value()) return std::nullopt;
  FunctionInfo info = opt_info.value();
  MACAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info->arg_types.size(), info->launch_param_tags);
  return PackFuncPackedArgAligned(f, info->arg_types);
}

ffi::Module MACAModuleCreate(ffi::Bytes data, ffi::String fmt,
                             ffi::Map<ffi::String, FunctionInfo> fmap, ffi::String maca_source) {
  auto n = ffi::make_object<MACAModuleNode>(data, fmt, fmap, maca_source);
  return ffi::Module(n);
}

ffi::Module MACAModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return MACAModuleCreate(ffi::Bytes(std::move(data)), ffi::String(fmt), fmap, ffi::String());
}

ffi::Module MACAModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  ffi::Bytes data;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::String fmt;
  stream.Read(&fmt);
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&data);
  return MACAModuleCreate(std::move(data), std::move(fmt), std::move(fmap), ffi::String());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.Module.load_from_file.maca", MACAModuleLoadFile)
      .def("ffi.Module.load_from_bytes.maca", MACAModuleLoadFromBytes)
      .def("ffi.Module.create.maca",
           [](ffi::Bytes data, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
              ffi::String maca_source) {
             return MACAModuleCreate(std::move(data), std::move(fmt), std::move(fmap),
                                     std::move(maca_source));
           });
}
}  // namespace runtime
}  // namespace tvm
