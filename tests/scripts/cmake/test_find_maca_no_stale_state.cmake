if(NOT TVM_SOURCE_DIR)
  get_filename_component(TVM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
endif()

include("${TVM_SOURCE_DIR}/cmake/utils/FindMACA.cmake")

set(MACA_FOUND TRUE)
set(MACA_INCLUDE_DIRS "/stale/include")
set(MACA_MACAMCC_LIBRARY "/stale/libmcruntime.so")

find_maca("${CMAKE_CURRENT_LIST_DIR}/does-not-exist")

if(MACA_FOUND)
  message(FATAL_ERROR "find_maca must clear stale MACA_FOUND when no SDK is found")
endif()

if(MACA_INCLUDE_DIRS OR MACA_MACAMCC_LIBRARY)
  message(FATAL_ERROR "find_maca must clear stale MACA include and library values")
endif()
