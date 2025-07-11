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
#include <tvm/ffi/function.h>

#include <array>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
#include "../meta_data.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "maca_common.h"

namespace tvm {
namespace runtime {

// Module to support thread-safe multi-GPU execution.
// mcModule_t is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class MACAModuleNode : public runtime::ModuleNode {
 public:
  explicit MACAModuleNode(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string maca_source)
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

  const char* type_key() const final { return "mc"; }
  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  }
  ffi::Function GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final;

  void SaveToFile(const String& file_name, const String& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "maca") {
      ICHECK_NE(maca_source_.length(), 0);
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, maca_source_);
    } else {
      ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, data_);
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

  String GetSource(const String& format) final {
    if (format == fmt_) {
      return data_;
    }
    if (format == "c" || format == "") {
      return maca_source_;
    }
    return "";
  }

  // get a CUfunction from primary context in device_id
  mcFunction_t GetFunc(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope

    if (module_[device_id] == nullptr) {
      MACA_DRIVER_CALL(mcModuleLoadData(&(module_[device_id]), data_.c_str()));
    }
    mcFunction_t func;
    mcError_t result = mcModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != mcSuccess) {
      LOG(FATAL) << "MACAError: mcModuleGetFunction " << func_name
                 << " failed with error: " << mcGetErrorString(result);
    }
    return func;
  }
  // get a global var from primary context in device_id
  mcDeviceptr_t GetGlobal(int device_id, const std::string& global_name, size_t expect_nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      MACA_DRIVER_CALL(mcModuleLoadData(&(module_[device_id]), data_.c_str()));
    }
    mcDeviceptr_t global = nullptr;
    size_t nbytes = 0;

    MACA_DRIVER_CALL(mcModuleGetGlobal(&global, &nbytes, module_[device_id], global_name.c_str()));
    ICHECK_EQ(nbytes, expect_nbytes);
    return global;
  }

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The maca source.
  std::string maca_source_;
  // the internal modules per GPU, to be lazily initialized.
  std::array<mcModule_t, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class MACAWrappedFunc {
 public:
  // initialize the MACA function.
  void Init(MACAModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
            size_t num_void_args, const std::vector<std::string>& launch_param_tags) {
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

    mcStream_t strm = static_cast<mcStream_t>(MACAThreadEntry::ThreadLocal()->stream);

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
  ObjectPtr<Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<mcFunction_t, kMaxNumGPUs> fcache_;
  // launch parameters configuration
  LaunchParamConfig launch_param_config_;
};

ffi::Function MACAModuleNode::GetFunction(const String& name,
                                          const ObjectPtr<Object>& sptr_to_self) {
  ICHECK_EQ(sptr_to_self.get(), this);
  ICHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return ffi::Function();
  const FunctionInfo& info = it->second;
  MACAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types.size(), info.launch_param_tags);
  return PackFuncPackedArgAligned(f, info.arg_types);
}

Module MACAModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap,
                        std::string maca_source) {
  auto n = make_object<MACAModuleNode>(data, fmt, fmap, maca_source);
  return Module(n);
}

Module MACAModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return MACAModuleCreate(data, fmt, fmap, std::string());
}

Module MACAModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return MACAModuleCreate(data, fmt, fmap, std::string());
}

TVM_FFI_REGISTER_GLOBAL("runtime.module.loadbinary_mcir").set_body_typed(MACAModuleLoadBinary);

TVM_FFI_REGISTER_GLOBAL("runtime.module.loadbinary_maca").set_body_typed(MACAModuleLoadBinary);
TVM_FFI_REGISTER_GLOBAL("runtime.module.loadbinary_mc").set_body_typed(MACAModuleLoadBinary);

TVM_FFI_REGISTER_GLOBAL("runtime.module.loadfile_mcir").set_body_typed(MACAModuleLoadFile);
TVM_FFI_REGISTER_GLOBAL("runtime.module.loadfile_maca").set_body_typed(MACAModuleLoadFile);
TVM_FFI_REGISTER_GLOBAL("runtime.module.loadfile_mcbin").set_body_typed(MACAModuleLoadFile);
}  // namespace runtime
}  // namespace tvm
