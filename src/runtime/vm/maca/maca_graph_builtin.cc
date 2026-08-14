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
 * \file src/runtime/vm/maca/maca_graph_builtin.cc
 * \brief The MACA graph related builtin functions for Relax virtual machine.
 */

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/vm/vm.h>

#include "../../../backend/maca/runtime/maca_common.h"
#include "../../../support/utils.h"
#include "dlpack/dlpack.h"
namespace tvm {
namespace runtime {
namespace vm {

namespace {

struct MACAGraphCaptureKey {
  // The unique index of the capture function within the module
  int64_t index;
  // The symbolic variables the capture function depends on. When the capture function is ran with
  // different symbolic variable values, the MACA graph will be re-captured as a different version,
  // identified by this shape tuple. This is default constructed as an empty tuple.
  ffi::Shape shape_expr;

  MACAGraphCaptureKey(int64_t index, const ffi::Optional<ffi::Shape>& shape_expr) : index(index) {
    if (shape_expr) {
      this->shape_expr = shape_expr.value();
    }
  }
};

struct MACAGraphCaptureKeyHash {
  size_t operator()(const MACAGraphCaptureKey& key) const {
    std::hash<int64_t> hash_fn;
    size_t hash = hash_fn(key.index);
    for (const auto& shape : key.shape_expr) {
      support::HashCombine(hash, hash_fn(shape));
    }
    return hash;
  }
};

struct MACAGraphCaptureKeyEqual {
  bool operator()(const MACAGraphCaptureKey& lhs, const MACAGraphCaptureKey& rhs) const {
    return lhs.index == rhs.index && std::equal(lhs.shape_expr.begin(), lhs.shape_expr.end(),
                                                rhs.shape_expr.begin(), rhs.shape_expr.end());
  }
};

/*! \brief The captured state of a MACA graph */
struct MACAGraphCapturedState {
  MACAGraphCapturedState() {}

  MACAGraphCapturedState(const MACAGraphCapturedState&) = delete;
  MACAGraphCapturedState(MACAGraphCapturedState&& other) { *this = std::move(other); }

  MACAGraphCapturedState& operator=(const MACAGraphCapturedState&) = delete;
  MACAGraphCapturedState& operator=(MACAGraphCapturedState&& other) {
    std::swap(states, other.states);
    std::swap(exec, other.exec);
    return *this;
  }

  ~MACAGraphCapturedState() {
    if (exec) {
      MACA_CALL(mcGraphExecDestroy(exec));
    }
  }

  /*!
   * \brief Tuple of intemediate tensors in the capture func that will be used outside the
   * capture func
   */
  ffi::ObjectRef states;
  /*! \brief The instantiated maca graph */
  mcGraphExec_t exec = nullptr;
};

class ScopedMACAStream {
 public:
  ScopedMACAStream() { MACA_CALL(mcStreamCreate(&stream_)); }
  ~ScopedMACAStream() { mcStreamDestroy(stream_); }
  ScopedMACAStream(const ScopedMACAStream&) = delete;
  ScopedMACAStream(ScopedMACAStream&&) = delete;
  ScopedMACAStream& operator=(const ScopedMACAStream&) = delete;
  ScopedMACAStream& operator=(ScopedMACAStream&&) = delete;

  operator mcStream_t() const { return stream_; }

 private:
  mcStream_t stream_;
};

class MACACaptureStream {
 public:
  explicit MACACaptureStream(mcGraph_t* graph) : output_graph_(graph) {
    MACA_CALL(mcGetDevice(&device_id_));
    TVM_FFI_CHECK_SAFE_CALL(
        TVMFFIEnvSetStream(kDLMACA, device_id_, capture_stream_,
                           reinterpret_cast<TVMFFIStreamHandle*>(&prev_default_stream_)));
    MACA_CALL(mcStreamBeginCapture(capture_stream_, mcStreamCaptureModeGlobal));
  }
  ~MACACaptureStream() noexcept(false) {
    mcError_t capture_error = mcStreamEndCapture(capture_stream_, output_graph_);
    if (capture_error != mcSuccess) {
      // The capture may have been invalidated by the exception that is
      // currently unwinding the stack.  Do not throw a second exception from
      // this destructor, but clear MACA's thread-local error so that it is not
      // reported by an unrelated MACA call later in the same host thread.
      mcGetLastError();
    }
    TVM_FFI_CHECK_SAFE_CALL(TVMFFIEnvSetStream(kDLMACA, device_id_, prev_default_stream_, nullptr));
  }

 private:
  int device_id_;
  mcStream_t prev_default_stream_;
  ScopedMACAStream capture_stream_;

  mcGraph_t* output_graph_;
};

}  // namespace

/*! \brief The VM extension of MACA graph. */
class MACAGraphExtensionNode : public VMExtensionNode {
 public:
  /*!
   * \brief Launch the maca graph if it has been cached, otherwise execute it in capture mode.
   * \param vm The virtual machine.
   * \param capture_func The function of type (args...) -> Tuple[ffi::ObjectRef], where 'args' are
   * the static arguments that are the same for all invocations of the capture function, the
   * returned tuple contains the intermediate tensors that will be used outside the capture
   * function.
   * \param args The static arguments of the capture function
   * \param entry_index The unique index of the capture function used for lookup.
   * \return The return value of the capture function.
   */
  ffi::ObjectRef RunOrCapture(VirtualMachine* vm, const ffi::ObjectRef& capture_func, Any args,
                              int64_t entry_index, ffi::Optional<ffi::Shape> shape_expr) {
    MACAGraphCaptureKey entry_key{entry_index, shape_expr};
    if (auto it = capture_cache_.find(entry_key); it != capture_cache_.end()) {
      // Launch MACA graph
      const auto& [states, exec] = it->second;
      int device_id;
      MACA_CALL(mcGetDevice(&device_id));
      MACA_CALL(
          mcGraphLaunch(exec, static_cast<mcStream_t>(TVMFFIEnvGetStream(kDLMACA, device_id))));
      return states;
    }

    // Set up arguments for the graph execution
    ffi::Array<Any> tuple_args = args.cast<ffi::Array<Any>>();
    int nargs = static_cast<int>(tuple_args.size());

    std::vector<AnyView> packed_args(nargs);
    for (int i = 0; i < nargs; ++i) {
      packed_args[i] = tuple_args[i];
    }

    ffi::Any capture_func_rv;
    // Run the function without MACA graph. This is a warm up step to do necessary initialization
    // of the MACA module such as loading module data, setting kernel attributes.
    vm->InvokeClosurePacked(capture_func, ffi::PackedArgs(packed_args.data(), nargs),
                            &capture_func_rv);

    // Run the graph in capture mode
    mcGraph_t graph;

    {
      MACACaptureStream capture_stream(&graph);
      vm->InvokeClosurePacked(capture_func, ffi::PackedArgs(packed_args.data(), nargs),
                              &capture_func_rv);
    }

    MACAGraphCapturedState entry;
    entry.states = capture_func_rv.cast<ffi::ObjectRef>();
    MACA_CALL(mcGraphInstantiate(&entry.exec, graph, NULL, NULL, 0));
    MACA_CALL(mcGraphDestroy(graph));

    ffi::ObjectRef states = entry.states;

    capture_cache_[entry_key] = std::move(entry);

    return states;
  }

  /*!
   * \brief Get the cached allocation from the cache or run the allocation function.
   * \param vm The virtual machine.
   * \param alloc_func The function of type () -> ffi::ObjectRef, where the returned object is the
   * tuple of allocated storage objects.
   * \param entry_index The unique index of the allocation function used for lookup.
   */
  ffi::ObjectRef GetCachedAllocation(VirtualMachine* vm, const ffi::ObjectRef& alloc_func,
                                     int64_t entry_index) {
    if (auto it = alloc_cache_.find(entry_index); it != alloc_cache_.end()) {
      return it->second;
    }
    ffi::Any alloc_func_rv;
    vm->InvokeClosurePacked(alloc_func, ffi::PackedArgs(nullptr, 0), &alloc_func_rv);
    ffi::ObjectRef alloc_result = alloc_func_rv.cast<ffi::ObjectRef>();
    alloc_cache_[entry_index] = alloc_result;
    return alloc_result;
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("vm.MACAGraphExtension", MACAGraphExtensionNode,
                                    VMExtensionNode);

 private:
  /*!
   * \brief The cache of captured maca graphs. The key is a unique index for the capture function.
   * The value is the result of the capture.
   */
  std::unordered_map<MACAGraphCaptureKey, MACAGraphCapturedState, MACAGraphCaptureKeyHash,
                     MACAGraphCaptureKeyEqual>
      capture_cache_;
  /*!
   * \brief The cache of allocations. The key is a unique index for the allocation function.
   * The value is the cached allocations, which is a tuple of storages.
   */
  std::unordered_map<int64_t, ffi::ObjectRef> alloc_cache_;
};

/*! Managed reference to MACAGraphExtensionNode */
class MACAGraphExtension : public VMExtension {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(MACAGraphExtension, VMExtension,
                                             MACAGraphExtensionNode);
  static MACAGraphExtension Create() {
    auto data_ = ffi::make_object<MACAGraphExtensionNode>();
    return MACAGraphExtension(std::move(data_));
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("vm.builtin.cuda_graph.run_or_capture",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    TVM_FFI_ICHECK(args.size() == 5 || args.size() == 4);
                    VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
                    auto extension = vm->GetOrCreateExtension<MACAGraphExtension>();
                    auto capture_func = args[1].cast<ffi::ObjectRef>();
                    Any func_args = args[2];
                    int64_t entry_index = args[3].cast<int64_t>();
                    ffi::Optional<ffi::Shape> shape_expr = std::nullopt;
                    if (args.size() == 5) {
                      shape_expr = args[4].cast<ffi::Shape>();
                    }
                    *rv = extension->RunOrCapture(vm, capture_func, func_args, entry_index,
                                                  shape_expr);
                  })
      .def_packed("vm.builtin.cuda_graph.get_cached_alloc", [](ffi::PackedArgs args, ffi::Any* rv) {
        TVM_FFI_ICHECK_EQ(args.size(), 3);
        VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
        auto extension = vm->GetOrCreateExtension<MACAGraphExtension>();
        auto alloc_func = args[1].cast<ffi::ObjectRef>();
        int64_t entry_index = args[2].cast<int64_t>();
        *rv = extension->GetCachedAllocation(vm, alloc_func, entry_index);
      });
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
